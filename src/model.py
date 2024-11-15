from typing import Optional, Union, Callable, Tuple
import math
from time import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.functional import scaled_dot_product_attention
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
        
    def forward(self, query: Tensor, attn_mask: Tensor, sin, cos) -> Tuple[Tensor, Optional[Tensor]]:
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

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    logger = logging.getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Tensor, sin, cos) -> Tensor:

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, sin=sin, cos=cos)
        if self.norm is not None:
            output = self.norm(output)

        return output
    
def expand_param(param: torch.Tensor, dim: int, size: int):
    mean = torch.mean(param, dim=dim, keepdim=True)
    std = torch.std(param, dim=dim, keepdim=True)
    added_shape = list(param.shape)
    added_shape[dim] = size - added_shape[dim]
    added_param = torch.randn(*added_shape, device=param.device, dtype=param.dtype)*std+mean
    param = torch.cat([param, added_param], dim=dim)
    return param

class Model(TransformerEncoder):
    logger = logging.getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, num_layers, d_model, nhead, d_ff_factor, dropout, activation, norm: bool, 
                vocs: list, padding_idx: int):
        layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*d_ff_factor,
            dropout=dropout, activation=activation,
            norm_first=True
        )
        if norm:
            norm = nn.LayerNorm(d_model ,elementwise_affine=False)
        else:
            norm = None
        num_embeddings = len(vocs)
        super().__init__(layer, num_layers=num_layers, norm=norm)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx)
        self.predictor = nn.Linear(d_model, num_embeddings)
        self.head_dim = d_model // nhead
        self.vocs = vocs

        def save_vocs(module, state_dict, prefix, local_metadata):
            state_dict[prefix+'vocs'] = self.vocs
        self._register_state_dict_hook(save_vocs)

        def load_pre_hook(module, state_dict, prefix, local_metadata, 
                strict, missing_keys, unexpected_keys, error_msgs) -> None:
            
            # match embedding
            state_vocs = np.array(state_dict[prefix+'vocs'], dtype=object)
            self_vocs = np.array(self.vocs, dtype=object)
            if np.any(state_vocs != self_vocs):
                self.logger.warning(f"vocs in state_dict does not match current vocs."
                        "Some weights will be permuted.")
                self.logger.debug(f"Removed from state_dict: {sorted(set(state_vocs)-set(self.vocs))}")
                self.logger.debug(f"New in model: {sorted(set(self.vocs)-set(state_vocs))}")
                common_vocs = list(set(list(state_vocs))&set(list(self_vocs)))
                state_idx = np.array([np.where(state_vocs == v)[0][0] for v in common_vocs])
                self_idx = np.array([np.where(self_vocs == v)[0][0] for v in common_vocs])
                for key in ['embedding.weight', 'predictor.weight', 'predictor.bias']:
                    state_param = state_dict[prefix+key]
                    size = list(state_param.shape)
                    size[0] = len(self_vocs)
                    mean = torch.mean(state_param, dim=0, keepdim=True)
                    std = torch.std(state_param, dim=0, keepdim=True)
                    new_param = torch.randn(*size, dtype=state_param.dtype, device=state_param.device)*std+mean
                    new_param[self_idx] = state_param[state_idx]
                    state_dict[prefix+key] = new_param
            
            # remove vocs
            del state_dict[prefix+'vocs']
        self._register_load_state_dict_pre_hook(load_pre_hook, with_module=True)

    def forward(self, src: torch.Tensor):
        x = self.embedding(src)
        length = x.shape[0]
        mask = nn.Transformer.generate_square_subsequent_mask(length).to(x.device).to(x.dtype)

        # rotate
        # rstart = time()
        position_enc = np.array([[pos / np.power(10000, 2 * j / self.head_dim) for j in range(self.head_dim//2)] 
                for pos in range(len(src))])
        sin = torch.tensor(np.sin(position_enc), device=x.device, dtype=x.dtype)
        cos = torch.tensor(np.cos(position_enc), device=x.device, dtype=x.dtype)
        # rend = time()
        # print(rstart - rend)

        x = super().forward(x, mask, sin, cos)
        return self.predictor(x)
    
    def generate(self, start_voc: str, end_voc: str, max_len: int, batch_size: int) -> torch.Tensor:
        self.logger.info("generate used")
        device = self.predictor.weight.device
        assert start_voc in self.vocs and end_voc in self.vocs
        vocs = np.array(self.vocs)
        start_token = np.where(vocs == start_voc)[0][0]
        end_token = np.where(vocs == end_voc)[0][0]

        is_finished = torch.full((batch_size,), fill_value=False, device=device)
        input = torch.full((1, batch_size), fill_value=start_token, 
            dtype=torch.long, device=device) # [L, B]

        for i in range(max_len):
            
            output = self(input) # [L, B, D]

            prob = F.softmax(output[-1], dim=1) # [B, D]
            output = torch.multinomial(prob, num_samples=1) # [B, 1]
            output = output.view(1, -1) # [1, B]
            is_finished = torch.logical_or(is_finished, output[0] == end_token)
            input = torch.cat([input, output], dim=0) # [L, B]
            if torch.all(is_finished): break
        return input.T

    def generate2(self, start_voc: str, end_voc: str, max_len: int, batch_size: int) -> torch.Tensor:
        self.logger.info("generate2 used")
        device = self.predictor.weight.device
        assert start_voc in self.vocs and end_voc in self.vocs
        vocs = np.array(self.vocs)
        start_token = np.where(vocs == start_voc)[0][0]
        end_token = np.where(vocs == end_voc)[0][0]

        is_finished = torch.full((batch_size,), fill_value=False, device=device)
        input = torch.full((1, batch_size), fill_value=start_token, 
            dtype=torch.long, device=device) # [L, B]
        attn_outputs = [
            torch.zeros((0, batch_size, 768), device=device, dtype=torch.float)
            for i_layer in range(len(self.layers))
        ]
        for i in range(max_len):
            
            # output = self(input) # [L, B, D]
            src = input

            x = self.embedding(src)
            length = x.shape[0]
            mask = nn.Transformer.generate_square_subsequent_mask(length).to(x.device).to(x.dtype)

            position_enc = np.array([[pos / np.power(10000, 2 * j / self.head_dim) for j in range(self.head_dim//2)] 
                    for pos in range(len(src))])
            
            sin = torch.tensor(np.sin(position_enc), device=x.device, dtype=x.dtype)
            cos = torch.tensor(np.cos(position_enc), device=x.device, dtype=x.dtype)

            mod: TransformerEncoderLayer
            for i_layer, mod in enumerate(self.layers):
                src_mask = mask
                assert mod.norm_first # norm_first=False is not supported
                xr = x
                x = mod.norm1(x)
                
                query = x
                num_heads = mod.self_attn.num_heads
                in_proj_weight = mod.self_attn.in_proj_weight
                in_proj_bias = mod.self_attn.in_proj_bias
                dropout_p = mod.self_attn.dropout
                out_proj_weight = mod.self_attn.out_proj.weight
                out_proj_bias = mod.self_attn.out_proj.bias
                training=mod.self_attn.training
                attn_mask=src_mask
                
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
                q = q[:,:,-1:]
                attn_output = scaled_dot_product_attention(q, k, v, dropout_p = dropout_p if training else 0.0)
                
                attn_output = attn_output.permute(2, 0, 1, 3).contiguous()[-1:].view(bsz * 1, embed_dim)
                
                attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
                attn_output = attn_output.view(1, bsz, attn_output.size(1))
                attn_output = torch.cat([attn_outputs[i_layer], attn_output[-1:]], dim=0)
                attn_outputs[i_layer] = attn_output
                x = attn_output

                x = mod.dropout1(x)
                x = xr + x

                xr = x
                x = mod.norm2(x)
                x = mod.linear2(mod.dropout(mod.activation(mod.linear1(x))))
                x = mod.dropout2(x)
                x = xr + x

            if self.norm is not None:
                x = self.norm(x)
            
            
            
            output = self.predictor(x)

            prob = F.softmax(output[-1], dim=1) # [B, D]
            output = torch.multinomial(prob, num_samples=1) # [B, 1]
            output = output.view(1, -1) # [1, B]
            is_finished = torch.logical_or(is_finished, output[0] == end_token)
            input = torch.cat([input, output], dim=0) # [L, B]
            if torch.all(is_finished): break
        return input.T
