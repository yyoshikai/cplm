from pathlib import Path
import yaml
import torch.nn as nn
import torch
from torch import Tensor
from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaForCausalLM

class MambaModel(MambaForCausalLM):
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
    
    def forward(self, src: Tensor):
        """
        src: [L, B]
        """
        output = super().forward(src.T.contiguous(), )
        x: Tensor = output['logits'] # [B, L, D]
        return x.transpose(0, 1) # [L, B, D]
        