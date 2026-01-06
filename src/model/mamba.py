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
from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaForCausalLM, MambaCache
from transformers.generation.streamers import BaseStreamer
from .transformer import save_vocs, align_embedding, right_to_left_padding
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

    def forward(self, src: Tensor, position: Tensor,
                get_mem: bool=False, offset: list[float]=None, mem_path: str=None):
        """
        src: [L, B]
        position: [L, B]
        """
        L, B = src.shape

        # check position
        assert torch.all((position == torch.arange(L).reshape(L, 1))|(src == self.config.pad_token_id)), \
            "Only sequential position is supported for Mamba"

        output = self.model(src.T.contiguous(), )
        x: Tensor = output['logits'] # [B, L, D]
        x = x.transpose(0, 1) # [L, B, D]

        # get_mem
        if get_mem:
            return tuple([x]+get_mems(src.device, offset, mem_path))
        else:
            return x

    @torch.no_grad()
    def generate(self, context: torch.Tensor, position: torch.Tensor,
                end_voc: str, max_len: int, pad_token: int, tqdm: bool=True, do_sample: bool=True):
        """
        context: Tensor[L, B](long)
        """
        assert self.model.config.eos_token_id == self.vocs.index(end_voc)
        assert self.model.config.pad_token_id == pad_token
        Lc, B = context.shape
        context = right_to_left_padding(context, pad_token)
        context = context.T # [B, L]

        # check position is sequential
        position_is_sequential = position == torch.arange(len(position)).unsqueeze(-1)
        position_is_sequential[:Lc] |= context == pad_token
        assert torch.all(position_is_sequential)

        device = next(self.parameters()).device
        self.logger.debug(f"GPU[pred]={self.get_gpuuse(B, Lc, False)/2**30:.03f}GB")
        self.logger.debug(f"{Lc=}")
        if tqdm:
            torch.cuda.reset_peak_memory_stats(device)

        # Left to right padding
        
        streamer = ProgressStreamer('tqdm', max_len, self) if tqdm else None
        outputs = self.model.generate(context, do_sample=do_sample, max_new_tokens=max_len, streamer=streamer) # [B, L]
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
    def __init__(self, name, max_len, model: MambaModel):
        self.pbar = tqdm(total=max_len, desc=name, miniters=1)
        self.model = model
        self.device = next(self.model.parameters()).device

    def put(self, value: torch.Tensor):
        if value.dim() == 1:
            mem = psutil.virtual_memory()

            used = torch.cuda.memory_allocated(self.device)
            used_max = torch.cuda.max_memory_allocated(self.device)
            GB = 2**30
            self.pbar.set_postfix_str(f"CPU mem:{mem.used/GB:.02f}/{mem.total/GB:.02f}GB, GPU mem: used={used/GB:.02f}, max={used_max/GB:.02f}", refresh=False)
            self.pbar.update(1)
        
    def end(self):
        pass
