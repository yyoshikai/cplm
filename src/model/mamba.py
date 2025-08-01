import sys, psutil, gc, copy
from pathlib import Path
from functools import partial
from typing import Optional
from logging import getLogger
from tqdm import tqdm
import yaml
import torch.nn as nn
import torch
from torch import Tensor
from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaForCausalLM, MambaCache
from transformers.generation.streamers import BaseStreamer
from .transformer import save_vocs, align_embedding
from src.utils import set_random_seed


class MambaModel(MambaForCausalLM):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, vocs: list, padding_idx: int, end_token: str):
        config = MambaConfig(
            vocab_size=len(vocs), 
        )
        with open(str(Path(__file__).parent / "configs" / "mamba-130m-hf.yaml")) as f:
            config = yaml.safe_load(f)
        config = MambaConfig(**config)
        config.vocab_size = len(vocs)
        config.pad_token_id = padding_idx
        config.eos_token_id = vocs.index(end_token)
        super().__init__(config)
        self._register_state_dict_hook(save_vocs)
        self._register_load_state_dict_pre_hook(partial(align_embedding, 
                embedding_name='backbone.embeddings', predictor_name='lm_head'), with_module=True)
        self.vocs = vocs

    def forward(self, src: Tensor):
        """
        src: [L, B]
        """
        output = super().forward(src.T.contiguous(), )
        x: Tensor = output['logits'] # [B, L, D]
        return x.transpose(0, 1) # [L, B, D]


class MambaModel2(nn.Module):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, vocs: list, padding_idx: int, end_token: str):
        super().__init__()

        # Build mamba model
        config = MambaConfig(
            vocab_size=len(vocs), 
        )
        with open(str(Path(__file__).parent / "configs" / "mamba-130m-hf.yaml")) as f:
            config = yaml.safe_load(f)
        config = MambaConfig(**config)
        config.vocab_size = len(vocs)
        config.pad_token_id = padding_idx
        config.eos_token_id = vocs.index(end_token)
        self.model = MambaForCausalLM(config)

        # Add hooks
        self._register_state_dict_hook(save_vocs)
        self._register_load_state_dict_pre_hook(partial(align_embedding, 
                embedding_name='model.backbone.embeddings', predictor_name='model.lm_head'), with_module=True)
        self.vocs = vocs

    def forward(self, src: Tensor):
        """
        src: [L, B]
        """
        output = self.model(src.T.contiguous(), )
        x: Tensor = output['logits'] # [B, L, D]
        return x.transpose(0, 1) # [L, B, D]
    
    def generate2(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int, remove_freq: int, tqdm=True) -> list[torch.Tensor]:
        """
        context: [L, B]
        """
        assert self.model.config.pad_token_id == pad_token
        assert self.model.config.eos_token_id == self.vocs.index(end_voc)
        output = []
        for i, item in enumerate(context.T):
            if pad_token in item:
                item = item[:torch.where(item == pad_token)[0][0]]
            output.append(self.model.generate(item.unsqueeze(0), do_sample=True, max_new_tokens=max_len, streamer=ProgressStreamer(str(i), max_len, self) if tqdm else None)[0])
            gc.collect() # TODO: 必要？
        return output
    
    @torch.no_grad()
    def generate(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int, remove_freq: int, tqdm: bool=True, do_sample: bool=True):
        
        assert self.model.config.eos_token_id == self.vocs.index(end_voc)
        assert self.model.config.pad_token_id == pad_token

        max_new_tokens = max_len
        
        streamer=ProgressStreamer("", max_new_tokens, self) if tqdm else None
        
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.update(**{'max_new_tokens': max_new_tokens, 'do_sample': do_sample})

        pad_token_id = generation_config.pad_token_id
        context = context.T
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
        if streamer is not None:
            streamer.put(input_ids.cpu())
        generation_config.max_length = generation_config.max_new_tokens + input_ids.shape[1]
        
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
                next_tokens = []
                for i in range(batch_size):
                    set_random_seed(cur_len)
                    probs = nn.functional.softmax(next_token_scores[i], dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(0)
                    next_tokens.append(next_token)
                next_tokens = torch.stack(next_tokens, dim=0)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            if cur_len < context_len:
                cur_prompt = context[:, cur_len]
                next_tokens[cur_prompt != pad_token_id] = cur_prompt[cur_prompt != pad_token_id]


            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            is_done = input_ids.shape[1] >= generation_config.max_length
            unfinished_sequences = unfinished_sequences & ~(
                torch.isin(input_ids[:, -1], generation_config._eos_token_tensor) | 
                torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)
            )

            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs
        if streamer is not None:
            streamer.end()
        outputs = []
        for i in range(4):
            input_ids0 = input_ids[i]
            input_ids0 = input_ids0[:torch.where(input_ids0 == generation_config.eos_token_id)[0][0]+1]
            outputs.append(input_ids0)
        return outputs

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