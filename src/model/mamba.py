from pathlib import Path
from functools import partial
from logging import getLogger
import yaml
import torch.nn as nn
import torch
from torch import Tensor
from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaForCausalLM
from transformers.generation.streamers import BaseStreamer
from .transformer import save_vocs, align_embedding
from ..utils.logger import INFO_WORKER
from time import time
import os


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
    
    def generate2(self, context: torch.Tensor, end_voc: str, max_len: int, pad_token: int, remove_freq: int, tqdm=True, result_dir=None, step=None, rank=None) -> list[torch.Tensor]:
        """
        context: [L, B]
        """
        assert self.model.config.pad_token_id == pad_token
        assert self.model.config.eos_token_id == self.vocs.index(end_voc)
        output = []
        log_dir = f"{result_dir}/gen_memories/{step}/{rank}"
        os.makedirs(log_dir, exist_ok=True)
        for i, item in enumerate(context.T):
            if pad_token in item:
                item = item[:torch.where(item == pad_token)[0][0]]
            output.append(self.model.generate(item.unsqueeze(0), do_sample=True, max_new_tokens=max_len, streamer=ProgressStreamer(str(i), f"{log_dir}/{i}.txt") if tqdm else None)[0])
        return output

class ProgressStreamer(BaseStreamer):
    logger = getLogger(__module__)

    def __init__(self, name, memory_path):
        self.name = name
        self.count = 0
        self.init = True
        self.memory_path = memory_path
        mstat = torch.cuda.memory_stats()
        with open(self.memory_path, 'w') as f:
            f.write('l,time,'+','.join(mstat.keys())+'\n')
        pass

    def put(self, value: torch.Tensor):
        l = value.shape[1] if value.dim() == 2 else value.shape[0]
        self.count += l
        
        # Log start
        if self.init:
            self.logger.log(INFO_WORKER, f"Generation {self.name}: started generation.")
        
        # Log l
        if self.count % 10 == 0 or self.init:
            self.logger.log(INFO_WORKER, f"Generation {self.name}: generated {self.count} tokens")

        # memory stat
        mstat = torch.cuda.memory_stats()
        with open(self.memory_path, 'a') as f:
            f.write(f"{l},{time()},"+','.join(map(str, mstat.values()))+'\n')
        self.init = False
    def end(self):
        self.logger.log(INFO_WORKER, f'Generation {self.name}: finished. ({self.count}) tokens.')
