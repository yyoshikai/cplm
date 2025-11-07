import logging
from typing import Optional, Union, Callable
from collections import defaultdict
from collections.abc import Iterable
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
from torch.nn.modules.transformer import _get_activation_fn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.parallel import DistributedDataParallel

from ..utils.memory import get_mems

def multi_head_attention_forward(
    query: Tensor,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    sin: Tensor, cos: Tensor,
    training: bool = True,
    attn_mask: Tensor = None,
) -> Tensor:
    
    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = query.shape
    head_dim = embed_dim // num_heads

    proj = F.linear(query, in_proj_weight, in_proj_bias)
    proj = proj.unflatten(-1, (3, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
    q, k, v = proj[0], proj[1], proj[2]
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    q = q.view(bsz, num_heads, tgt_len, head_dim)
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)

    sin = sin[:src_len]
    cos = cos[:src_len]
    sin_pos = torch.stack([sin, sin], dim=-1).reshape(src_len, head_dim)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape(src_len, head_dim)

    rotate_half_q = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
    q = q * cos_pos + rotate_half_q * sin_pos
    rotate_half_k = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k)
    k = k * cos_pos + rotate_half_k * sin_pos
    attn_output = scaled_dot_product_attention(q, k, v, dropout_p = dropout_p if training else 0.0, is_causal=True)
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
    return attn_output

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
        
    def forward(self, query: Tensor, attn_mask: Tensor, sin, cos) -> tuple[Tensor, Optional[Tensor]]:
        return multi_head_attention_forward(
            query, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            sin, cos,
            training=self.training,
            attn_mask=attn_mask)

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

    def forward(self, src: Tensor, src_mask: Tensor, sin, cos) -> Tensor:

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, sin, cos)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, sin, cos))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Tensor, sin, cos) -> Tensor:
        x = self.self_attn(x, attn_mask=attn_mask, sin=sin, cos=cos)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def generate_init_cache(self, batch_size, device: torch.device=None) -> dict[str, torch.Tensor]:
        if device is None: device = self.linear2.weight.device
        return {
            'k': torch.zeros((batch_size, self.self_attn.num_heads, 0, self.self_attn.head_dim), 
                    device=device, dtype=self.linear2.weight.dtype),
            'v': torch.zeros((batch_size, self.self_attn.num_heads, 0, self.self_attn.head_dim), 
                    device=device, dtype=self.linear2.weight.dtype),
        }
    
    def slice_cache(self, cache, indices) -> dict:
        return { 'k': cache['k'][indices], 'v': cache['v'][indices] }

    def forward_one(self, x, cache, sin, cos):
        assert self.norm_first # norm_first=False is not supported
        xr = x
        x = self.norm1(x)
        
        num_heads = self.self_attn.num_heads
        in_proj_weight = self.self_attn.in_proj_weight
        in_proj_bias = self.self_attn.in_proj_bias
        dropout_p = self.self_attn.dropout
        out_proj_weight = self.self_attn.out_proj.weight
        out_proj_bias = self.self_attn.out_proj.bias
        training=self.self_attn.training
        
        # set up shape vars
        _, bsz, embed_dim = x.shape
        head_dim = embed_dim // num_heads

        proj = F.linear(x, in_proj_weight, in_proj_bias)
        proj = proj.unflatten(-1, (3, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = proj[0], proj[1], proj[2]

        q = q.view(1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        q = q.view(bsz, num_heads, 1, head_dim)
        k = k.view(bsz, num_heads, 1, head_dim)
        v = v.view(bsz, num_heads, 1, head_dim)
        v = torch.cat([cache['v'], v], dim=2)
        cache['v'] = v

        sin_pos = torch.stack([sin, sin], dim=-1).reshape(1, head_dim)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape(1, head_dim)

        rotate_half_q = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
        q = q * cos_pos + rotate_half_q * sin_pos
        rotate_half_k = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k)
        k = k * cos_pos + rotate_half_k * sin_pos

        k = torch.cat([cache['k'], k], dim=2)
        cache['k'] = k

        attn_output = scaled_dot_product_attention(q, k, v, dropout_p = dropout_p if training else 0.0)
        
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(1*bsz, embed_dim)
        
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(1, bsz, attn_output.size(1))
        x = attn_output

        x = self.dropout1(x)
        x = xr + x

        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
        return x, cache

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

        self.pos_buffer_len = None
        sin, cos, mask = self.make_pos_buffer(pos_buffer_len)
        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('mask', mask, persistent=False)
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(self, ['sin', 'cos', 'mask'])        
        self.pos_buffer_len = pos_buffer_len

        # Log # of make_pos_buffer
        self.n_make_pos_buffer = 0
        self.n_forward = 0


        self._register_state_dict_hook(save_vocs)
        self._register_load_state_dict_pre_hook(partial(align_embedding, 
                embedding_name='embedding', predictor_name='predictor'), with_module=True)
        
        self.gpuuse_coef = lru_cache(1)(self._gpuuse_coef)
        self.get_capture_rate = lru_cache(1)(self._get_capture_rate)
        

    def make_pos_buffer(self, length):
        position_enc = np.array([[pos / np.power(10000, 2 * j / self.head_dim) for j in range(self.head_dim//2)] 
                for pos in range(length)])
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        sin = torch.tensor(np.sin(position_enc), device=device, dtype=dtype)
        cos = torch.tensor(np.cos(position_enc), device=device, dtype=dtype)
        mask = nn.Transformer.generate_square_subsequent_mask(length).to(device).to(dtype)
        return sin, cos, mask
        

    def forward(self, src: torch.Tensor, get_mem: bool=False, offset: list[float]=None, mem_path: str=None):
        self.n_forward += 1
        x = self.embedding(src)
        L = x.shape[0]
        if L > self.pos_buffer_len:
            sin, cos, mask = self.make_pos_buffer(L)
            self.n_make_pos_buffer += 1
        else:
            sin, cos, mask = self.sin[:L], self.cos[:L], self.mask[:L, :L]
        if (self.n_forward&(self.n_forward-1)) == 0:
            self.logger.debug(f"make_pos_buffer call={self.n_make_pos_buffer}/{self.n_forward}")
        output = x
        for layer in self.layers:
            output = layer(output, src_mask=mask, sin=sin, cos=cos)
        if self.norm is not None:
            output = self.norm(output)
        x = output
        x = self.predictor(x)

        # get_mem
        if get_mem:
            return tuple([x]+get_mems(src.device, offset, mem_path))
        else:
            return x

    def generate2(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int, remove_freq: int, tqdm=True) -> list[torch.Tensor]:
        """
        Use kv-cache, remove finished samples

        context: torch.Tensor(long)[L, B]
        """
        
        device = self.predictor.weight.device
        vocs = np.array(self.vocs)
        end_token = np.where(vocs == end_voc)[0][0]
        L, B = context.shape
        assert L >= 1
        Ngen = L+max_len-1
        """
        ex. If max_len = 5, L = 1:
        generate [START], *, *, *, *, * -> 5 times
        """

        finished_idxs = []
        finished_outputs = []
        finished_ns_generated = []
        context_sizes = torch.sum(context != pad_token, dim=0)
        indices = torch.arange(B, dtype=torch.int, device=device) # [B,]
        is_finished = torch.full((B,), fill_value=False, device=device) # [B,]
        ns_generated = torch.zeros(B, dtype=torch.int, device=device)
        input = context[:1] # [1, B]
        cache = [layer.generate_init_cache(B) for layer in self.layers]
        outputs = input # [1, B]

        sin_buf, cos_buf, mask_buf = (self.sin, self.cos, self.mask) \
            if Ngen <= self.pos_buffer_len else self.make_pos_buffer(Ngen)
        del mask_buf

        poss_pbar = _tqdm(range(Ngen)) if tqdm else range(Ngen)
        for pos in poss_pbar:

            x = self.embedding(input)

            layer: TransformerEncoderLayer
            for i_layer, layer in enumerate(self.layers):
                x, cache[i_layer] = layer.forward_one(x, cache[i_layer], sin_buf[pos:pos+1], cos_buf[pos:pos+1])

            if self.norm is not None:
                x = self.norm(x)

            x = self.predictor(x)
            prob = F.softmax(x[0], dim=1) # [B, D]
            output = torch.multinomial(prob, num_samples=1) # [B, 1]
            output = output.view(-1) # [B]
            if pos < L-1:
                pos_context = context[pos+1]
                is_in_context = pos_context != pad_token
                output[is_in_context] = pos_context[is_in_context]
            else:
                is_in_context = torch.full_like(output, fill_value=False, dtype=torch.bool)
                
            end_token_generated = (~is_in_context)&(output == end_token)
            ns_generated[(~is_in_context)&(~is_finished)] += 1
            outputs = torch.cat([outputs, output.unsqueeze(0)], dim=0) # [L, B]

            is_finished = is_finished|end_token_generated|(ns_generated >= max_len)
            input = output.unsqueeze(0)
            if torch.all(is_finished): break

            # Remove finished entries from inference
            if (pos+1) % remove_freq == 0:
                finished_js = torch.where(is_finished)[0]
                remain_js = torch.where(~is_finished)[0]
                finished_idxs += indices[finished_js].tolist()
                finished_outputs += list(outputs[:, finished_js].T)
                finished_ns_generated += ns_generated[finished_js].tolist()

                indices = indices[remain_js]
                is_finished = is_finished[remain_js]
                ns_generated = ns_generated[remain_js]
                input = input[:, remain_js]
                cache = [layer.slice_cache(c, remain_js) for c, layer in zip(cache, self.layers)]
                outputs = outputs[:, remain_js]
                context = context[:, remain_js]

                if tqdm:
                    poss_pbar.set_postfix_str(f"{len(remain_js)}/{B} remains, {is_in_context.sum()} in context")
        finished_idxs += indices.tolist()
        finished_outputs += list(outputs.T)
        finished_ns_generated += ns_generated.tolist()
            
        # Order outputs
        finished_idxs_inv = np.argsort(finished_idxs)
        outputs = [finished_outputs[idx] for idx in finished_idxs_inv]
        ns_generated = [finished_ns_generated[idx] for idx in finished_idxs_inv]

        # cut outputs
        outputs = [output[context_size:context_size+n_generated] 
                for output, context_size, n_generated in zip(outputs, context_sizes, ns_generated)]
        return outputs

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

