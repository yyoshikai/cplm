from pathlib import Path
from functools import partial
from logging import getLogger
import yaml
import torch.nn as nn
import torch
from torch import Tensor
from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaForCausalLM
from .transformer import save_vocs, align_embedding

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

