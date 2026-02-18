import math, warnings
from pathlib import Path
from functools import partial, lru_cache
from collections import defaultdict
from typing import Optional
from logging import getLogger
from tqdm import tqdm

import yaml
import pandas as pd
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaForCausalLM, MambaCache
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from .transformer import save_vocs, align_embedding
from .model import Streamer, Model
from ..utils.memory import get_mems

# This warning is inherent in mamba of the version we used.
warnings.filterwarnings('ignore', message="torch.get_autocast_gpu_dtype() is deprecated.", category=DeprecationWarning, module='mamba_ssm.ops.selective_scan_interface', lineno=201)

class MambaModel(Model):
    """
    Contents in ./gpuuse are from /workspace/resource_test/240921_transformer_size
    """
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, vocs: list, padding_idx: int, end_token: str, **kwargs):
        super().__init__()

        # Build mamba model
        with open(str(Path(__file__).parent / "configs" / "mamba-130m-hf.yaml")) as f:
            config = yaml.safe_load(f)

        # Other parameters are not supported due to gpu_use()
        supported_keys = {'num_hidden_layers', 'hidden_size', 'state_size', 'intermediate_size', 'time_step_rank', 'conv_kernel'}
        keys = set(kwargs.keys())
        if len(keys - supported_keys) > 0:
            raise ValueError(f"Following kwargs are not supported: {keys - supported_keys}")

        config.update(kwargs)
        config = MambaConfig(**config)
        config.vocab_size = len(vocs)
        config.pad_token_id = padding_idx
        config.eos_token_id = vocs.index(end_token)
        self.config = config
        self.model = MambaForCausalLM(config)

        # Add hooks
        self._register_state_dict_hook(save_vocs)
        self._register_load_state_dict_pre_hook(partial(align_embedding, 
                embedding_name='model.backbone.embeddings', predictor_name='model.lm_head'), with_module=True)
        self.vocs = vocs

        self.gpuuse_coef = lru_cache(1)(self._gpuuse_coef)
        self.get_capture_rate = lru_cache(1)(self._get_capture_rate)

    def forward(self, src: Tensor, position: Tensor, out_state: bool=False,
                get_mem: bool=False, offset: list[float]=None, mem_path: str=None):
        """
        src: [L, B]
        position: [L, B]
        """
        L, B = src.shape

        # check position
        assert torch.all((position == torch.arange(L, device=position.device).reshape(L, 1))|(src == self.config.pad_token_id)), \
            "Only sequential position is supported for Mamba"

        model_output = self.model(src.T.contiguous(), output_hidden_states=True)
        x: Tensor = model_output['logits'] # [B, L, D]
        x = x.transpose(0, 1) # [L, B, D]

        # get_mem
        output = (x,)
        if out_state:
            output = output+(model_output['hidden_states'][-1].transpose(0, 1),) # [L, B, D]
        if get_mem:
            output = output+tuple(get_mems(src.device, offset, mem_path))
        return output[0] if len(output) == 1 else output

    @torch.inference_mode()
    def generate2(self, contexts: list[Tensor], positions: list[list[int]], streamers: list[Streamer], max_new_token: int|None):
        """
        contexts: list[Tensor(L, torch.long)]
        positions: list[Tensor(L, torch.long)]
        """
        # get shape
        device = next(self.parameters()).device

        for position in positions:
            assert torch.all(torch.tensor(position, device=device) == torch.arange(len(position), device=device))

        wrapper = StreamerWrapper(streamers)
        
        criteria = StoppingCriteriaList()
        criteria.append(StreamerCriteria(wrapper))

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: Tensor):
            return wrapper.outs[batch_id][2]
        
        if max_new_token is None:
            max_new_token = math.inf

        # Left to right padding
        contexts = pad_sequence_left(contexts).T # [B, L]
        self.model.generate(contexts, stopping_criteria = criteria, streamer=wrapper, max_new_tokens=max_new_token, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, do_sample=True) # [B, L]

    def prepare_inputs_for_generation(self, 
            input_ids,
            cache_params: Optional[MambaCache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            **kwargs,
    ):
        if cache_position[0] > 0:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = None
        else:
            cache_position = torch.arange(0, self.model.config.conv_kernel, device=input_ids.device)

        model_inputs = {
                "input_ids": input_ids.contiguous(),
                "cache_params": cache_params,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
            }
        return model_inputs
    
    def _gpuuse_coef(self, bf16: bool) -> dict[tuple[int, int], float]:
        """
        
        Returns
        -------
        dim2coefs: dict[tuple[int, int], float]
        dim2coefs[batch_size_dim, legnth_dim] = coef of memory use
        """

        # calc param sizes
        ## params
        state_size = self.config.state_size
        num_hidden_layers = self.config.num_hidden_layers
        intermediate_size = self.config.intermediate_size
        hidden_size = self.config.hidden_size
        voc_size = self.config.vocab_size
        time_step_rank = self.config.time_step_rank
        conv_kernel = self.config.conv_kernel

        ## shape
        mname = 'mamba_bf16' if bf16 else 'mamba'
        t2dim2coefs = {}
        for t in ['forward', 'backward']:
            dfp = pd.read_csv(Path(__file__).parent / "gpuuse" / t / (mname+'.csv'), keep_default_na=False)
            dim2coefs = defaultdict(float)
            for shape, itemsize, n in zip(dfp['shape'], dfp['itemsize'], dfp['n']):
                batch_size_dim = length_dim = ceiled_length_dim = 0
                coef = float(eval(n)) * itemsize
                shape = shape.split(' ') if len(shape) > 0 else []
                for d in shape:
                    if d == 'batch_size':
                        batch_size_dim += 1
                    elif d == 'length':
                        length_dim += 1
                    elif d == 'math.ceil(length/2048)':
                        ceiled_length_dim += 1
                    else:
                        try:
                            coef = coef*eval(d)
                        except Exception as e:
                            print(f"invalid d: {d}")
                            raise e
                dim2coefs[batch_size_dim, length_dim, ceiled_length_dim] += coef
            t2dim2coefs[t] = dict(dim2coefs)
        return t2dim2coefs
    
    def _get_capture_rate(self, bf16):
        ## capture rate
        with open(str(Path(__file__).parent / "gpuuse" / "capture_rates.yaml")) as f:
            capture_rates = yaml.safe_load(f)
        mname = 'mamba_bf16' if bf16 else 'mamba'
        return capture_rates[mname]
        
    def get_gpuuse(self, batch_size: int, length: int, bf16: bool, kernel: str|None=None, capture_rate: bool=True):
        t2dim2coefs = self.gpuuse_coef(bf16)
        max_gpuuse = 0
        for dim2coefs in t2dim2coefs.values():
            gpuuse = 0
            ceiled_length = math.ceil(length / 2048)
            for (batch_size_dim, length_dim, ceiled_length_dim), coef in dim2coefs.items():
                gpuuse += (batch_size**batch_size_dim) \
                        * (length**length_dim) \
                        * (ceiled_length**ceiled_length_dim) \
                        * coef
            max_gpuuse = max(gpuuse, max_gpuuse)
        if capture_rate:
            max_gpuuse = max_gpuuse / self.get_capture_rate(bf16)
        return max_gpuuse

    @property
    def state_size(self):
        return self.config.hidden_size

class StreamerWrapper(BaseStreamer):
    def __init__(self, streamers: list[Streamer]):
        self.streamers = streamers
        self.outs: list[tuple[bool, list[int], list[int]]|None] = [None for _ in streamers]
    def put(self, value: torch.Tensor):
        if value.dim() == 1:
            value.unsqueeze_(-1) # [B, 1(L)]
        for b, streamer in enumerate(self.streamers):
            if self.outs[b] is None or self.outs[b][0]:
                v = value[b]
                v = v[v != 0]
                self.outs[b] = streamer.put(v.tolist())

    def end(self):
        pass

class StreamerCriteria(StoppingCriteria):
    def __init__(self, streamer_wrapper: StreamerWrapper):
        self.wrapper = streamer_wrapper
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        return ~torch.tensor([out is None or out[0] for out in self.wrapper.outs], device=input_ids.device, dtype=torch.bool)

def pad_sequence_left(sequences: list[Tensor], padding_value: float=0) -> Tensor:
    return pad_sequence([s.flip(0) for s in sequences], padding_value=padding_value).flip(0).contiguous()