import psutil, copy, math
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
from .transformer import save_vocs, align_embedding
from ..utils.memory import get_mems

class MambaModel(nn.Module):
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

    def forward(self, src: Tensor, get_mem: bool=False, offset: list[float]=None, mem_path: str=None):
        """
        src: [L, B]
        """
        output = self.model(src.T.contiguous(), )
        x: Tensor = output['logits'] # [B, L, D]
        x = x.transpose(0, 1) # [L, B, D]

        # get_mem
        if get_mem:
            return tuple([x]+get_mems(src.device, offset, mem_path))
        else:
            return x

    @torch.no_grad()    
    def generate2(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int, remove_freq: int, tqdm: bool=True, do_sample: bool=True):
        """
        context: Tensor[L, B](long)
        """
        assert self.model.config.eos_token_id == self.vocs.index(end_voc)
        assert self.model.config.pad_token_id == pad_token
        Lc, B = context.shape

        # Left to right padding
        is_pad = (context != pad_token).to(torch.int)
        if torch.any(is_pad[:-1] > is_pad[1:]):
            self.logger.warning(f"context is modified to left-padding.")
            c_revs = list(context.flip(0).T) # [B][L]
            c_revs = [c[c != pad_token] for c in c_revs]
            context_rev = pad_sequence(c_revs, padding_value=pad_token, batch_first=True) # [B, L]
            context = context_rev.flip(1)
        
        outputs = self.model.generate(context, do_sample=True, max_new_tokens=max_len) # [B, L]
        generateds = outputs[:, Lc:]
        
        # truncate
        eos_token_id = self.model.config.eos_token_id
        generateds = [
            g[:torch.where(g == eos_token_id)[0][0]+1] 
            if eos_token_id in g else g
            for g in generateds ]
        return generateds

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
        
    def get_gpuuse(self, batch_size: int, length: int, bf16: bool, kernel: str=None, capture_rate: bool=True):
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

class ProgressStreamer(BaseStreamer):
    def __init__(self, name, max_len, model):
        self.pbar = tqdm(total=max_len, desc=name, miniters=1)
        self.model = model

    def put(self, value: torch.Tensor):
        l = value.shape[1] if value.dim() == 2 else 1
        mem = psutil.virtual_memory()
        GB = 2**30
        self.pbar.set_postfix_str(f"mem={mem.used/GB:.03f}/{mem.total/GB:.03f}GB", refresh=False)
        self.pbar.update(l)
        
    def end(self):
        pass
