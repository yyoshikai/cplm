from typing import Optional, Union, Callable
from collections.abc import Iterable
from functools import partial
from tqdm import tqdm as _tqdm
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.parallel import DistributedDataParallel
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
        if predictor.bias: keys.append(f'{predictor_name}.bias')
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
    __constants__ = ['norm']
    def __init__(self, num_layers, d_model, nhead, d_ff_factor, dropout, 
            activation, norm: bool, vocs: list, padding_idx: int, pos_buffer_len: int=100):
        if norm:
            norm = nn.LayerNorm(d_model ,elementwise_affine=False)
        else:
            norm = None
        num_embeddings = len(vocs)
        
        super().__init__()
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
        self.make_pos_buffer(pos_buffer_len)

        self._register_state_dict_hook(save_vocs)
        self._register_load_state_dict_pre_hook(partial(align_embedding, 
                embedding_name='embedding', predictor_name='predictor'), with_module=True)

    def make_pos_buffer(self, length):
        if self.pos_buffer_len is not None:
            self.logger.info(f"Length of positional buffers will be reset to {length}.")
        position_enc = np.array([[pos / np.power(10000, 2 * j / self.head_dim) for j in range(self.head_dim//2)] 
                for pos in range(length)])
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        sin = torch.tensor(np.sin(position_enc), device=device, dtype=dtype)
        cos = torch.tensor(np.cos(position_enc), device=device, dtype=dtype)
        mask = nn.Transformer.generate_square_subsequent_mask(length).to(device).to(dtype)
        
        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('mask', mask, persistent=False)
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(self, ['sin', 'cos', 'mask'])
        
        self.pos_buffer_len = length

    def forward(self, src: torch.Tensor):
        x = self.embedding(src)
        L = x.shape[0]
        if L > self.pos_buffer_len:
            self.make_pos_buffer(L)
        output = x
        for layer in self.layers:
            output = layer(output, src_mask=self.mask[:L, :L], sin=self.sin[:L], cos=self.cos[:L])
        if self.norm is not None:
            output = self.norm(output)
        x = output

        return self.predictor(x)
    
    def generate(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int) -> torch.Tensor:
        """
        Use kv-cache for generation

        context: torch.Tensor(long)[L, B]
        """
        
        device = self.predictor.weight.device
        vocs = np.array(self.vocs)
        end_token = np.where(vocs == end_voc)[0][0]
        context_len, batch_size = context.shape
        assert context_len >= 1

        if max_len > self.pos_buffer_len:
            self.make_pos_buffer(max_len)

        is_finished = torch.full((batch_size,), fill_value=False, device=device)
        input = context[:1] # [1, B]
        cache = [layer.generate_init_cache(batch_size) for layer in self.layers]
        outputs = [input.squeeze(0)]
        for pos in _tqdm(range(max_len), dynamic_ncols=True):

            x = self.embedding(input)

            sin = self.sin[pos:pos+1]
            cos = self.cos[pos:pos+1]

            layer: TransformerEncoderLayer
            for i_layer, layer in enumerate(self.layers):
                x, cache[i_layer] = layer.forward_one(x, cache[i_layer], sin, cos)

            if self.norm is not None:
                x = self.norm(x)

            x = self.predictor(x)
            prob = F.softmax(x[0], dim=1) # [B, D]
            output = torch.multinomial(prob, num_samples=1) # [B, 1]
            output = output.view(-1) # [B]
            if pos < context_len-1:
                pos_context = context[pos+1]
                output[pos_context != pad_token] = pos_context[pos_context != pad_token]
            outputs.append(output)
                
            is_finished = torch.logical_or(is_finished, output == end_token)
            input = output.unsqueeze(0)
            if torch.all(is_finished): break
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def generate2(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int, remove_freq: int, tqdm=True) -> list[torch.Tensor]:
        """
        Use kv-cache, remove finished samples

        context: torch.Tensor(long)[L, B]
        """
        
        device = self.predictor.weight.device
        vocs = np.array(self.vocs)
        end_token = np.where(vocs == end_voc)[0][0]
        context_len, batch_size = context.shape
        assert context_len >= 1

        finished_idxs = []
        finished_outputs = []

        indices = torch.arange(batch_size, dtype=torch.int, device=device) # [B,]
        is_finished = torch.full((batch_size,), fill_value=False, device=device) # [B,]
        input = context[:1] # [1, B]
        cache = [layer.generate_init_cache(batch_size) for layer in self.layers]
        outputs = input # [1, B]

        if max_len > self.pos_buffer_len:
            self.make_pos_buffer(max_len)

        for pos in _tqdm(range(max_len)) if tqdm else range(max_len):

            x = self.embedding(input)

            layer: TransformerEncoderLayer
            for i_layer, layer in enumerate(self.layers):
                x, cache[i_layer] = layer.forward_one(x, cache[i_layer], self.sin[pos:pos+1], self.cos[pos:pos+1])

            if self.norm is not None:
                x = self.norm(x)

            x = self.predictor(x)
            prob = F.softmax(x[0], dim=1) # [B, D]
            output = torch.multinomial(prob, num_samples=1) # [B, 1]
            output = output.view(-1) # [B]
            if pos < context_len-1:
                pos_context = context[pos+1]
                output[pos_context != pad_token] = pos_context[pos_context != pad_token]
            outputs = torch.cat([outputs, output.unsqueeze(0)], dim=0) # [L, B]
                
            is_finished = torch.logical_or(is_finished, output == end_token)
            input = output.unsqueeze(0)
            if torch.all(is_finished): break

            # Remove finished entries from inference
            if (pos+1) % remove_freq == 0:
                finished_js = torch.where(is_finished)[0]
                remain_js = torch.where(~is_finished)[0]
                finished_idxs += indices[finished_js].tolist()
                finished_outputs += list(outputs[:, finished_js].T)

                
                indices = indices[remain_js]
                is_finished = is_finished[remain_js]
                input = input[:, remain_js]
                cache = [layer.slice_cache(c, remain_js) for c, layer in zip(cache, self.layers)]
                outputs = outputs[:, remain_js]
                context = context[:, remain_js]
        finished_idxs += indices.tolist()
        finished_outputs += list(outputs.T)
            
        # Order outputs
        finished_idxs_inv = np.argsort(finished_idxs)
        outputs = [finished_outputs[idx] for idx in finished_idxs_inv]
        return outputs

    def generate3(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int, tokens_per_batch: int, rebatch_lens: Iterable[int]) -> list[torch.Tensor]:
        """
        Use kv-cache, dynamic batching

        context: torch.Tensor(long)[L, B]

        遅かった。
        """

        assert tokens_per_batch >= max_len

        device = self.predictor.weight.device
        cpu = torch.device('cpu')

        vocs = np.array(self.vocs)
        end_token = np.where(vocs == end_voc)[0][0]

        context_len, batch_size = context.shape
        assert context_len >= 1
        rebatch_lens = sorted(list(rebatch_lens))
        assert rebatch_lens[-1] == max_len

        if max_len > self.pos_buffer_len:
            self.make_pos_buffer(max_len)
        
        finished_idxs = []
        finished_outputs = []

        all_context = context.cpu()
        all_indices = torch.arange(batch_size, dtype=torch.int, device=cpu) # [B,]
        all_input = context[:1] # [1, B]
        all_cache = [layer.generate_init_cache(batch_size, cpu) for layer in self.layers]
        all_outputs = all_input # [1, B]

        pbar = _tqdm(total=max_len)
        l_start = 0
        for l_end in rebatch_lens:
            cb_size = tokens_per_batch // l_end

            all_remain_js = []
            new_input = []
            new_cache = []
            new_outputs = []

            cb_total_step = 0
            ncb = len(range(0, len(all_indices), cb_size))
            for cb_start in range(0, len(all_indices), cb_size):
                cb_end = min(cb_start+cb_size, len(all_indices))

                context = all_context[:, cb_start:cb_end].to(device)
                input = all_input[:, cb_start:cb_end].to(device)
                cache = [{k: v[cb_start:cb_end].to(device) for k, v in c.items()}
                    for c in all_cache]
                outputs = [all_outputs[:, cb_start:cb_end]]
                is_finished = torch.full((cb_end-cb_start,), fill_value=False, dtype=torch.bool, device=device)

                for pos in range(l_start, l_end):

                    x = self.embedding(input)

                    layer: TransformerEncoderLayer
                    for i_layer, layer in enumerate(self.layers):
                        x, cache[i_layer] = layer.forward_one(x, cache[i_layer], self.sin[pos:pos+1], self.cos[pos:pos+1])

                    if self.norm is not None:
                        x = self.norm(x)

                    x = self.predictor(x)
                    prob = F.softmax(x[0], dim=1) # [B, D]
                    output = torch.multinomial(prob, num_samples=1) # [B, 1]
                    output = output.view(-1) # [B]
                    if pos < context_len-1:
                        pos_context = context[pos+1]
                        output[pos_context != pad_token] = pos_context[pos_context != pad_token]
                    is_finished = torch.logical_or(is_finished, output == end_token)
                    output = output.unsqueeze(0)
                    outputs.append(output.cpu())
                    input = output
                    cb_total_step += 1
                    if cb_total_step % ncb == 0:
                        pbar.update()
                outputs = torch.cat(outputs, dim=0) # [L, B]    
                
                finished_js = torch.where(is_finished)[0].cpu()
                remain_js = torch.where(~is_finished)[0].cpu()

                finished_idxs += all_indices[finished_js+cb_start].tolist()
                finished_outputs += list(outputs[:, finished_js].T.cpu())

                all_remain_js.append(remain_js.cpu()+cb_start)
                new_input.append(input[:, remain_js].cpu()) # [1, B]
                new_cache.append([{k: v[remain_js].cpu() for k, v in c.items()} 
                    for c in cache])
                new_outputs.append(output[:, remain_js].cpu())


            all_remain_js = torch.cat(all_remain_js)
            all_outputs = torch.cat(new_outputs, dim=1)
            if len(all_remain_js) == 0:
                break
            all_context = all_context[:, all_remain_js]
            all_indices = all_indices[all_remain_js]


            all_input = torch.cat(new_input, dim=1)
            all_cache = []
            for cs in zip(*new_cache):
                c = {key: torch.cat([c[key] for c in cs]) for key in cs[0].keys()}
                all_cache.append(c)
            l_start = l_end
        finished_idxs += all_remain_js.tolist()
        finished_outputs += list(all_outputs.T)

        # Order outputs
        outputs = [(idx, output) for idx, output in zip(finished_idxs, finished_outputs)]
        outputs.sort(key=lambda x: x[0])
        outputs = [output[1] for output in outputs]
        return outputs
