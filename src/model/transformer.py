import logging
import itertools as itr
from typing import Optional, Union, Callable
from collections import defaultdict
from functools import partial, lru_cache
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.nn.modules.transformer import _get_activation_fn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.parallel import DistributedDataParallel
from ..utils.memory import get_mems

logger = logging.getLogger(__name__)

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
        
        self.n_forward = 0

    def forward(self, x: Tensor, sin: Tensor, cos: Tensor, is_causal: bool=True, src_mask: Tensor|None=None, cache: dict[str, Tensor]|None=None) -> tuple[Tensor, Optional[Tensor]]:
        """
        x: Tensor[L, B, D](float)
        src_mask: Tensor expandable to [B, H, Lx, Lc]
        sin: Tensor[L, Dh/2](float) or [B, L, Dh]
        cos: Tensor[L, Dh/2](float) or [B, L, Dh]
        cache:
            'k': Tensor[B, H, L, Dh]
            'v': Tensor[B, H, L, Dh] 

        """
        self.n_forward += 1

        # set up shape vars
        L, B, D = x.shape
        Dh = D // self.num_heads

        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias) # [L, B, D*3]
        proj = proj.reshape(1, L, B, 3, D).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = proj[0], proj[1], proj[2] # [L, B, D]

        q = q.view(L, B * self.num_heads, Dh).transpose(0, 1).view(B, self.num_heads, L, Dh) # [B, H, L, Dh]
        k = k.view(L, B * self.num_heads, Dh).transpose(0, 1).view(B, self.num_heads, L, Dh) # [B, H, L, Dh]
        v = v.view(L, B * self.num_heads, Dh).transpose(0, 1).view(B, self.num_heads, L, Dh) # [B, H, L, Dh]
            
        sin_pos = torch.stack([sin, sin], dim=-1).reshape(-1, 1, L, Dh) # [1/B, 1, L, Dh]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape(-1, 1, L, Dh) # [1/B, 1, L, Dh]

        rotate_half_q = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
        q = q * cos_pos + rotate_half_q * sin_pos
        rotate_half_k = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k)
        k = k * cos_pos + rotate_half_k * sin_pos

        if cache is not None:
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)
        if src_mask is not None:
            src_mask = src_mask.contiguous()
        cache = {'k': k, 'v': v}
        attn_output = scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), dropout_p = self.dropout if self.training else 0.0, attn_mask=src_mask, is_causal=is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(B * L, D)

        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(L, B, attn_output.size(1))
        return attn_output, cache

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(self, src: Tensor, sin: Tensor, cos: Tensor, is_causal: bool, src_mask: Tensor|None=None, cache: dict[str, Tensor]|None=None) -> Tensor:

        x = src
        if self.norm_first:
            d_x, cache = self._sa_block(self.norm1(x), sin, cos, is_causal, src_mask, cache)
            x = x + d_x
            x = x + self._ff_block(self.norm2(x))
        else:
            d_x, cache = self._sa_block(x, sin, cos, is_causal, src_mask, cache)
            x = self.norm1(x + d_x)
            x = self.norm2(x + self._ff_block(x))
        return x, cache

    def _sa_block(self, x: Tensor, sin, cos, is_causal, src_mask, cache) -> Tensor:
        x, cache = self.self_attn(x, sin, cos, is_causal, src_mask, cache)
        return self.dropout1(x), cache

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def save_vocs(module, state_dict, prefix, local_metadata):
    state_dict[prefix+'vocs'] = module.vocs

def align_embedding(module: nn.Module, state_dict, prefix, local_metadata, 
        strict, missing_keys, unexpected_keys, error_msgs, 
        embedding_name, predictor_name) -> None:
    
    embedding = module.get_submodule(embedding_name)
    assert isinstance(embedding, nn.Embedding), \
            f"{embedding_name=} is incorrect ({type(embedding)})."
    predictor = module.get_submodule(predictor_name)
    assert isinstance(predictor, nn.Linear), \
            f"{predictor_name=} is incorrect ({type(predictor)})."

    # match embedding
    state_vocs = np.array(state_dict[prefix+'vocs'], dtype=object)
    module_vocs = np.array(module.vocs, dtype=object)
    if np.any(state_vocs != module_vocs):
        module.logger.warning(f"vocs in state_dict does not match current vocs. "
                "Some weights will be permuted.")
        module.logger.info(f"Removed from state_dict: {sorted(set(state_vocs)-set(module.vocs))}")
        module.logger.info(f"New in model: {sorted(set(module.vocs)-set(state_vocs))}")
        common_vocs = list(set(list(state_vocs))&set(list(module_vocs)))
        state_idx = np.array([np.where(state_vocs == v)[0][0] for v in common_vocs])
        self_idx = np.array([np.where(module_vocs == v)[0][0] for v in common_vocs])
        keys = [f'{embedding_name}.weight', f'{predictor_name}.weight']
        if predictor.bias is not None: keys.append(f'{predictor_name}.bias')
        for key in keys:
            state_param = state_dict[prefix+key]
            size = list(state_param.shape)
            size[0] = len(module_vocs)
            mean = torch.mean(state_param, dim=0, keepdim=True)
            std = torch.std(state_param, dim=0, keepdim=True)
            new_param = torch.randn(*size, dtype=state_param.dtype, device=state_param.device)*std+mean
            new_param[self_idx] = state_param[state_idx]
            state_dict[prefix+key] = new_param
    
    # remove vocs
    del state_dict[prefix+'vocs']

class Streamer:
    def put(self, tokens: list[int]):
        raise NotImplementedError

class Model(nn.Module):
    logger = logging.getLogger(f"{__module__}.{__qualname__}")
    data_logger = logging.getLogger(f"dexs.{__module__}.{__qualname__}")
    __constants__ = ['norm']
    def __init__(self, num_layers, d_model, nhead, d_ff_factor, dropout, 
            activation, norm: bool, vocs: list, padding_idx: int, pos_buffer_len: int=100):
        num_embeddings = len(vocs)
        
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff_factor = d_ff_factor
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*d_ff_factor,
            dropout=dropout, activation=activation,
            norm_first=True) for i in range(num_layers)])

        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model ,elementwise_affine=False) if norm else nn.Identity()
        
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx)
        self.predictor = nn.Linear(d_model, num_embeddings)
        self.head_dim = d_model // nhead
        self.vocs = vocs

        self.pos_buffer_len = 0
        self.n_make_pos_buffer = 0
        sin, cos = self.get_pos_buffer(pos_buffer_len)
        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)
        self.pos_buffer_len = pos_buffer_len
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(self, ['sin', 'cos']) 

        self.n_make_pos_buffer = 0 # reset here
        self.n_forward = 0

        self._register_state_dict_hook(save_vocs)
        self._register_load_state_dict_pre_hook(partial(align_embedding, 
                embedding_name='embedding', predictor_name='predictor'), with_module=True)
        
        self.gpuuse_coef = lru_cache(1)(self._gpuuse_coef)
        self.get_capture_rate = lru_cache(1)(self._get_capture_rate)
        
    
    def get_pos_buffer(self, size_or_positions: int|list[int]) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        sin: Tensor[L, Dh](float)
        cos: Tensor[L, Dh](float)
        """
        sup_pos = size_or_positions if isinstance(size_or_positions, int) else max(size_or_positions)+1
        if sup_pos <= self.pos_buffer_len: # fast path
            if isinstance(size_or_positions, int):
                return self.sin[:size_or_positions], self.cos[:size_or_positions]
            else:
                return self.sin[size_or_positions], self.cos[size_or_positions]
        self.n_make_pos_buffer += 1
        positions = range(sup_pos) if isinstance(size_or_positions, int) else size_or_positions
        position_enc = np.array([[pos / np.power(10000, 2 * j / self.head_dim) for j in range(self.head_dim//2)] 
                for pos in positions])
        sin = torch.tensor(np.sin(position_enc)).to(self.embedding.weight)
        cos = torch.tensor(np.cos(position_enc)).to(self.embedding.weight)
        return sin, cos
            
    def forward(self, src: Tensor, position: Tensor,
            get_mem: bool=False, offset: list[float]=None, mem_path: str=None, out_pos_buffer: bool=False):
        """
        src: [L, B]
        position: [L, B]
        """
        self.n_forward += 1
        x = self.embedding(src) # [L, B, D]
        L, B, D = x.shape
        Dh = self.head_dim
        sin, cos = self.get_pos_buffer(L) # [L, D]
        sin = torch.gather(
            input=sin.reshape(1, L, Dh//2).expand(B, L, Dh//2), dim=1,
            index=position.T.reshape(B, L, 1).expand(B, L, Dh//2)
        ).contiguous() # [B, L, D]
        cos = torch.gather(
            input=cos.reshape(1, L, Dh//2).expand(B, L, Dh//2), dim=1,
            index=position.T.reshape(B, L, 1).expand(B, L, Dh//2)
        ).contiguous() # [B, L, D]

        if (self.n_forward&(self.n_forward-1)) == 0:
            self.logger.debug(f"make_pos_buffer call={self.n_make_pos_buffer}/{self.n_forward}")
        for layer in self.layers:
            x, _ = layer(x, sin, cos, is_causal=True)
        x = self.predictor(self.norm(x))

        # get_mem
        output = (x, )
        if get_mem:
            output += tuple(get_mems(src.device, offset, mem_path))
        if out_pos_buffer:
            output += (sin, cos, )
        if len(output) == 1: output = output[0]
        return output

    @torch.inference_mode()
    def generate2(self, contexts: list[Tensor], positions: list[list[int]], streamers: list[Streamer], max_new_token: int|None, position_log_idxs: list[int]=[], token_range_log_idxs: list[int]=[]):
        """
        contexts: list[Tensor(L, torch.long)]
        positions: list[Tensor(L, torch.long)]
        """

        # get shape
        B = len(contexts)
        context_sizes = [len(context) for context in contexts]
        device = contexts[0].device
        do_position_logs = [idx in position_log_idxs for idx in range(len(contexts))]
        do_token_range_logs = [idx in token_range_log_idxs for idx in range(len(contexts))]

        # check shape
        assert len(streamers) == B
        assert max_new_token is None or max_new_token >= 1
        for context_size, position in zip(context_sizes, positions):
            assert len(position) == context_size

        # Initial forward
        caches = [{'k': [], 'v': []} for layer in self.layers] # caches[i_layer][k/v][i_data]
        cur_inputs = []
        is_continues = []
        next_positions = []
        for b in range(B):
            is_continue, next_position, next_token_range = streamers[b].put(contexts[b].tolist()) # [L], list[int]
            is_continues.append(is_continue)
            if not is_continue: continue
            if do_position_logs[b]:
                self.logger.debug(f"position[{b}][0]={positions[b]}")
            if do_token_range_logs[b]:
                self.logger.debug(f"next_token_range[{b}][0]={[self.vocs[v] for v in next_token_range]}")
            sin, cos = self.get_pos_buffer(positions[b]) # [L, Dh], [L, Dh]
            x = contexts[b].unsqueeze(-1) # [L, 1(B)]
            x = self.embedding(x)
            for il, layer in enumerate(self.layers):
                x, cache = layer(x, sin, cos, is_causal=True, cache=None) # [L, 1, D], {'k': [1, H, L, Dh]}
                for k, v in cache.items():
                    v = v.squeeze(0).transpose(0, 1) # [L, H, Dh]
                    caches[il][k].append(v.detach().clone())
            logit = self.predictor(self.norm(x[-1, 0])) # [L, 1, D] -> [D]
            range_logit = logit[next_token_range]
            range_output = torch.multinomial(F.softmax(range_logit, dim=0), num_samples=1).item()
            output = next_token_range[range_output]
            cur_inputs.append(output)
            next_positions.append(next_position)
        streamers = list(itr.compress(streamers, is_continues))
        cache = [{k: pad_sequence(v, batch_first=True).transpose(1, 2) for k, v in cache.items()} for cache in caches] # [i_layer][k,v][B, H, L, Dh]
        Bc, _, L, _ = cache[0]['k'].shape
        src_mask = torch.zeros((Bc, L), device=device) # [B, Lkv]
        for b, context in enumerate(contexts):
            src_mask[b, len(context):] = -torch.inf

        for i_gen in (range(1, max_new_token) if max_new_token is not None else itr.count(1)):
            positions = next_positions
            is_continues, next_positions, next_token_ranges = zip(*[ 
                streamer.put([cur_input]) for streamer, cur_input in zip(streamers, cur_inputs)
            ])
            if not any(is_continues):
                break
            for idx in np.where(do_position_logs)[0]:
                self.logger.debug(f"position[{idx}][{i_gen}]={positions[idx]}")
            for idx in np.where(do_token_range_logs)[0]:
                self.logger.debug(f"next_token_range[{idx}][{i_gen}]={[self.vocs[v] for v in next_token_ranges[idx]]}")

            # remove finished sample
            if not all(is_continues):
                cur_inputs, positions, next_positions, next_token_ranges, streamers, do_position_logs, do_token_range_logs \
                    = zip(*itr.compress(zip(cur_inputs, positions, next_positions, next_token_ranges, streamers, do_position_logs, do_token_range_logs), is_continues))
                is_continues = torch.tensor(is_continues, device=device)
                ## src_mask
                src_mask = src_mask[is_continues]
                length_mask = torch.any(src_mask == 0, dim=0) # [Lkv]
                src_mask = src_mask[:, length_mask]
                ## cache
                cache = [{k: v[is_continues][:, :, length_mask] for k, v in layer_cache.items()}
                        for layer_cache in cache]

            # make position
            sin, cos = self.get_pos_buffer([position[0] for position in positions]) # [B, Dh], [B, Dh]
            sin, cos = sin.unsqueeze(1), cos.unsqueeze(1) # [B, 1, Dh], # [B, 1, Dh]
            src_mask = torch.cat([src_mask, torch.zeros((len(src_mask), 1), device=device)], dim=-1)

            x = torch.tensor(cur_inputs, dtype=torch.long, device=device).unsqueeze(0) # [1(L), B]
            x = self.embedding(x)
            for il, layer in enumerate(self.layers):
                x, cache[il] = layer(x, sin, cos, is_causal=False, cache=cache[il], src_mask=src_mask.unsqueeze(1).unsqueeze(2))
            logits = self.predictor(self.norm(x[0])) # [1(L), B, D] -> [B, D]
            cur_inputs = []
            for b, (logit, next_token_range) in enumerate(zip(logits, next_token_ranges)):
                range_logit = logit[next_token_range]
                range_output = torch.multinomial(F.softmax(range_logit, dim=-1), num_samples=1).item()
                output = next_token_range[range_output]
                cur_inputs.append(output)

    def _get_mname(self, bf16: bool, kernel: str):        
        mname = f'tf_{kernel.lower()}'
        if bf16: mname += '_bf16'
        if kernel == 'FLASH' and (self.d_model // self.nhead) % 8 == 0:
            mname += '_8'
        return mname

    def _gpuuse_coef(self, bf16: bool, kernel: str) -> dict[tuple[int, int], float]:
        """
        
        Returns
        -------
        dim2coefs: dict[tuple[int, int], float]
        dim2coefs[batch_size_dim, legnth_dim] = coef of memory use
        """

        if kernel not in ['FLASH', 'EFFICIENT']:
            raise ValueError(f"Unsupported {kernel=}")
        if kernel == 'FLASH' and not bf16:
            raise ValueError("FLASH attention without bf16 is not supported.")

        # calc param sizes
        ## params
        d_model = self.d_model
        num_layers = self.num_layers
        nhead = self.nhead
        d_ff_factor = self.d_ff_factor
        voc_size = len(self.vocs)
        pos_buffer_len = self.pos_buffer_len

        ## mname
        mname = self._get_mname(bf16, kernel)

        ## shape
        t2dim2coefs = {}
        for t in ['forward', 'backward']:
            dfp = pd.read_csv(Path(__file__).parent / "gpuuse" / t / (mname+'.csv'), keep_default_na=False)
            dim2coefs = defaultdict(float)
            for shape, itemsize, n in zip(dfp['shape'], dfp['itemsize'], dfp['n']):
                batch_size_dim = length_dim = 0
                coef = float(eval(n)) * itemsize
                shape = shape.split(' ') if len(shape) > 0 else []
                for d in shape:
                    if d == 'batch_size':
                        batch_size_dim += 1
                    elif d == 'length':
                        length_dim += 1
                    else:
                        coef = coef*eval(d)
                
                dim2coefs[batch_size_dim, length_dim] += coef
            t2dim2coefs[t] = dict(dim2coefs)
        return t2dim2coefs
    
    def _get_capture_rate(self, bf16: bool, kernel: str):
        ## capture rate
        with open(str(Path(__file__).parent / "gpuuse" / "capture_rates.yaml")) as f:
            capture_rates = yaml.safe_load(f)
        return capture_rates[self._get_mname(bf16, kernel)]

    def get_gpuuse(self, batch_size: int, length: int, bf16: bool, kernel: str, 
            capture_rate: bool=True):

        t2dim2coefs = self.gpuuse_coef(bf16, kernel)
        max_gpuuse = 0
        for dim2coefs in t2dim2coefs.values():
            gpuuse = 0
            for (batch_size_dim, length_dim), coef in dim2coefs.items():
                gpuuse += (batch_size**batch_size_dim) * (length**length_dim) * coef
            max_gpuuse = max(gpuuse, max_gpuuse)
        if capture_rate:
            max_gpuuse = max_gpuuse / self.get_capture_rate(bf16=bf16, kernel=kernel)
        return max_gpuuse

def right_to_left_padding(inputs: Tensor, padding_idx: int) -> Tensor:
    return left_to_right_padding(inputs.flip(0), padding_idx, _is_right=True).flip(0)

def left_to_right_padding(inputs: Tensor, padding_idx: int, _is_right: bool=False) -> Tensor:
    """
    Parameters
    ----------
    inputs: Tensor[L, B]
    padding_idx: int        
    """
    is_context = (inputs != padding_idx).to(torch.int)
    if torch.any(is_context[1:] > is_context[:-1]):
        logger.warning(f"context is modified to {'right' if _is_right else 'left'}-padding.")
        is_ = list(inputs.T) # [B][L]
        is_ = [i[i != padding_idx] for i in is_]
        inputs = pad_sequence(is_, padding_value=padding_idx) # [L, B]
        inputs = inputs
    return inputs

