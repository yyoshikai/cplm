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
    def generate1(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int, remove_freq: int, tqdm: bool=True, do_sample: bool=True):
        """
        Parameters
        ----------
        context: torch.Tensor[L, B](long)
        
        
        """
        
        assert self.model.config.eos_token_id == self.vocs.index(end_voc)
        assert self.model.config.pad_token_id == pad_token

        max_new_tokens = max_len
        
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.update(**{'max_new_tokens': max_new_tokens, 'do_sample': do_sample})

        pad_token_id = generation_config.pad_token_id
        context = context.T # [B, L]
        batch_size, context_len = context.shape
        if pad_token_id in context:
            start_len = torch.where(torch.any(context == pad_token_id, dim=0))[0][0].item()
        else:
            start_len = context_len
        input_ids = context[:, :start_len]

        # Convert special tokens to tensors
        device = input_ids.device
        def _tensor_or_none(token, device=None):
            return token if token is None else torch.tensor(token, device=device, dtype=torch.long)
        generation_config._bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        generation_config._eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device).unsqueeze(0)
        generation_config._pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        generation_config._decoder_start_token_tensor = _tensor_or_none(generation_config.decoder_start_token_id, device=device)

        attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)
        generation_config.max_length = generation_config.max_new_tokens + context.shape[1]
        streamer=ProgressStreamer("", generation_config.max_length-1, self) if tqdm else None
        if streamer is not None:
            streamer.put(input_ids.cpu())
        
        # ---- sample ------

        # init values
        pad_token_id = generation_config._pad_token_tensor
        do_sample = generation_config.do_sample

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        cache_position = torch.ones(cur_len, dtype=torch.int64, device=device).cumsum(0) - 1
        cache_params = None

        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, 
                cache_params, cache_position, attention_mask)
            outputs = self.model(**model_inputs, use_cache=generation_config.use_cache, return_dict=True)
            
            cache_params = outputs.get("cache_params", None)
            cache_position = cache_position[-1:] + 1
            
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            top_k = min(generation_config.top_k, next_token_logits.size(-1))  # Safety check
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_scores = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            if cur_len < context_len:
                next_prompt = context[:, cur_len]
                next_tokens[next_prompt != pad_token_id] = next_prompt[next_prompt != pad_token_id]

            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            is_done = input_ids.shape[1] >= generation_config.max_length # 251018 多分合ってる
            if cur_len < context_len:
                end_token_generated = torch.isin(input_ids[:, -1], generation_config._eos_token_tensor) & (next_prompt == pad_token_id)
            else:
                end_token_generated = torch.isin(input_ids[:, -1], generation_config._eos_token_tensor)
            
            unfinished_sequences = unfinished_sequences & ~(
                end_token_generated | torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)
            )

            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs
        if streamer is not None:
            streamer.end()

        # truncate
        context_sizes = (context != pad_token_id).sum(dim=1)
        outputs = []
        eos_token_id = generation_config.eos_token_id
        for i in range(batch_size):
            output = input_ids[i]
            output = output[context_sizes[i]:]
            output = output[:max_len]
            if eos_token_id in output:
                output = output[:torch.where(output == eos_token_id)[0][0]+1]
            outputs.append(output)
        return outputs

    @torch.no_grad()    
    def generate2(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int, remove_freq: int, tqdm: bool=True, do_sample: bool=True):
        """
        context: Tensor[L, B](long)
        """
        assert self.model.config.eos_token_id == self.vocs.index(end_voc)
        assert self.model.config.pad_token_id == pad_token
        Lc, B = context.shape
        context = context.T # [B, L]

        device = next(self.parameters()).device
        self.logger.info(f"GPU[pred]={self.get_gpuuse(B, Lc, False)/2**30:.03f}GB")
        self.logger.info(f"{Lc=}")
        torch.cuda.reset_peak_memory_stats(device)

        # Left to right padding
        is_pad = (context != pad_token).to(torch.int)
        if torch.any(is_pad[:-1] > is_pad[1:]):
            self.logger.warning(f"context is modified to left-padding.")
            c_revs = list(context.flip(1)) # [B][L]
            c_revs = [c[c != pad_token] for c in c_revs]
            context_rev = pad_sequence(c_revs, padding_value=pad_token, batch_first=True) # [B, L]
            context = context_rev.flip(1)
        
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
