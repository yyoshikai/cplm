import logging
from typing import Optional, Union, Callable
from collections import defaultdict
from functools import partial, lru_cache
from tqdm import tqdm as _tqdm
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
        
        cache = {'k': k, 'v': v}
        attn_output = scaled_dot_product_attention(q, k, v, dropout_p = self.dropout if self.training else 0.0, attn_mask=src_mask, is_causal=is_causal)
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

class Model(nn.Module):
    logger = logging.getLogger(f"{__module__}.{__qualname__}")
    data_logger = logging.getLogger(f"dexs.{__module__}.{__qualname__}")
    __constants__ = ['norm']
    def __init__(self, num_layers, d_model, nhead, d_ff_factor, dropout, 
            activation, norm: bool, vocs: list, padding_idx: int, pos_buffer_len: int=100):
        if norm:
            norm = nn.LayerNorm(d_model ,elementwise_affine=False)
        else:
            norm = None
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
        self.norm = norm

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
        
    
    def get_pos_buffer(self, L: int, get_mask: bool=True) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        sin: Tensor[L, Dh](float)
        cos: Tensor[L, Dh](float)
        mask: Tensor[L, L](float)
        """
        if L > self.pos_buffer_len:
            self.n_make_pos_buffer += 1
            position_enc = np.array([[pos / np.power(10000, 2 * j / self.head_dim) for j in range(self.head_dim//2)] 
                    for pos in range(L)])
            device = self.embedding.weight.device
            dtype = self.embedding.weight.dtype
            sin = torch.tensor(np.sin(position_enc), device=device, dtype=dtype)
            cos = torch.tensor(np.cos(position_enc), device=device, dtype=dtype)
            return sin, cos
        else:
            return self.sin[:L], self.cos[:L]
            
    def forward(self, src: Tensor, position: Tensor,
            get_mem: bool=False, offset: list[float]=None, mem_path: str=None):
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
            input=cos.reshape(L, 1, Dh//2).expand(L, B, Dh//2), dim=1,
            index=position.reshape(L, B, 1).expand(L, B, Dh//2)
        ).contiguous() # [B, L, D]

        if (self.n_forward&(self.n_forward-1)) == 0:
            self.logger.debug(f"make_pos_buffer call={self.n_make_pos_buffer}/{self.n_forward}")
        output = x
        for layer in self.layers:
            output, _ = layer(output, sin, cos, is_causal=True)
        if self.norm is not None:
            output = self.norm(output)
        x = output
        x = self.predictor(x)

        # get_mem
        if get_mem:
            return tuple([x]+get_mems(src.device, offset, mem_path))
        else:
            return x

    @torch.inference_mode()
    def generate(self, context: torch.Tensor, position: torch.Tensor|None,
                end_voc: str, max_len: int, pad_token: int, tqdm=True, do_sample: bool=True) -> list[torch.Tensor]:
        """
        context: torch.Tensor(long)[L, B]
        position: torch.Tensor(long)[Lp, B]
        """
        assert do_sample # TODO

        Lc, B = context.shape
        if position is not None:
            Lp, _ = position.shape
        Dh = self.head_dim
        max_input_size = Lc+max_len-1
        device = context.device
        assert Lc >= 1
        if position is not None:
            assert Lp == max_input_size
        end_token = self.vocs.index(end_voc)

        context = left_to_right_padding(context, pad_token) # [L, B]
        context_sizes = torch.sum(context != pad_token, dim=0).tolist()
        sin_buf, cos_buf = self.get_pos_buffer(max_input_size, get_mask=False) # [L, Dh],  [L, Dh],  [L, Dh]
        if position is not None:
            sin_buf = torch.gather(
                input=sin_buf.reshape(1, max_input_size, Dh//2).expand(B, max_input_size, Dh//2), dim=1,
                index=position.T.reshape(B, max_input_size, 1).expand(B, max_input_size, Dh//2)
            ).contiguous() # [B, L, D]
            cos_buf = torch.gather(
                input=cos_buf.reshape(max_input_size, 1, Dh//2).expand(max_input_size, B, Dh//2), dim=1,
                index=position.reshape(max_input_size, B, 1).expand(max_input_size, B, Dh//2)
            ).contiguous() # [B, L, D]
            
        gen_pbar = _tqdm(range(max_len)) if tqdm else range(max_len)
        gen_iter = iter(gen_pbar)

        outputs = []
        is_ended = torch.full((B,), False, device=device)
        def forward_one(x: Tensor, is_ended: Tensor):
            """
            x: [B, Nt]
            """
            x = self.predictor(x)
            prob = F.softmax(x, dim=1) # [B, D]
            output = torch.multinomial(prob, num_samples=1).view(-1) # [B, 1] -> [B]
            outputs.append(output)
            is_ended = torch.logical_or(is_ended, output == end_token)
            next_input = output.unsqueeze(0) # [1, B]
            
            if tqdm:
                GB = 2**30
                g = torch.cuda.memory_allocated()
                g_max = torch.cuda.max_memory_allocated()
                gen_pbar.set_postfix_str(f"now={g/GB:.02f}GB, max={g_max/GB:.02f}GB")
            
            return next_input, is_ended

        # Initial forward
        cache_minis = [{'k': [], 'v': []} for layer in self.layers]
        cur_input_minis = []
        is_ended_minis = []
        i_gen = next(gen_iter)
        for b in (b_pbar:=_tqdm(range(0, B)) if tqdm else range(0, B)):
            context_size = context_sizes[b]
            x = self.embedding(context[:context_size,b:b+1])
            for i_layer, layer in enumerate(self.layers):    
                x, cache_mini_i = layer(x, sin_buf[...,:context_size,:], cos_buf[...,:context_size,:], is_causal=True, cache=None) # [L, B, D], {'k': [B, H, Lc, Dh]}
                for k, v in cache_mini_i.items():
                    # [1(B), H, Lc, Dh] -> [H, Lc, Dh] -> [Lc, H, Dh]
                    v = v.squeeze(0).transpose(0, 1)
                    cache_minis[i_layer][k].append(v.detach().clone()) # なぜか必要
            if self.norm is not None:
                x = self.norm(x)
            cur_input_mini, is_ended_mini = forward_one(x[-1], is_ended[b:b+1])
            cur_input_minis.append(cur_input_mini)
            is_ended_minis.append(is_ended_mini)
            if tqdm:
                GB = 2**30
                g = torch.cuda.memory_allocated()
                g_max = torch.cuda.max_memory_allocated()
                b_pbar.set_postfix_str(f"now={g/GB:.02f}GB, max={g_max/GB:.02f}GB")

        cache = [{k: pad_sequence(v, batch_first=True).transpose(1, 2) for k, v in cache_minis_i.items()} for cache_minis_i in cache_minis]
        cur_input = torch.cat(cur_input_minis, dim=1)
        is_ended = torch.cat(is_ended_minis, dim=0)
        outputs = [torch.cat(outputs, dim=0)]
        cache_size = Lc

        # make padded positional buffers
        sins = []
        coss = []
        for b, size in enumerate(context_sizes):
            pad_size = Lc - size
            sins.append(sin_buf[b,Lc-pad_size:Lc-pad_size+max_len-1,:])
            coss.append(cos_buf[b,Lc-pad_size:Lc-pad_size+max_len-1,:])
        sins = torch.stack(sins, dim=0) # [B, L, Dh]
        coss = torch.stack(coss, dim=0) # [B, L, Dh]
        src_mask_bool = torch.cat([
            context.T == pad_token, # [B, Lc]
            torch.full((B, max_len-1), False, dtype=torch.bool, device=device) # [B, max_len-1]
        ], dim=1).reshape(B, 1, 1, -1) # [B, 1, 1, L]
        src_masks = torch.zeros_like(src_mask_bool, dtype=sins.dtype)
        src_masks.masked_fill_(src_mask_bool, -torch.inf)

        # generation forward
        for i_gen in gen_iter:
            if torch.all(is_ended): break
            x = self.embedding(cur_input)
            for i_layer, layer in enumerate(self.layers):    
                x, cache[i_layer] = layer(x, 
                        sins[:, cache_size-Lc:cache_size-Lc+1],
                        coss[:, cache_size-Lc:cache_size-Lc+1], 
                        is_causal=False, 
                        cache=cache[i_layer], 
                        src_mask=src_masks[..., :cache_size+1]
                ) # [L, B, D], {'k': [B, H, L, Dh]}
            
            cur_input, is_ended = forward_one(x[0], is_ended)
            cache_size += 1
        outputs = torch.stack(outputs, dim=-1) # [B, L]

        # cut outputs
        cut_outputs = []
        for output in outputs:
            if end_token in output:
                output = output[:torch.where(output == end_token)[0][0]+1]
            cut_outputs.append(output)
        return cut_outputs        

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

