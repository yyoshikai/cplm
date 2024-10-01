from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, is_causal: Optional[bool] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=None, is_causal=is_causal, src_key_padding_mask=None)
        if self.norm is not None:
            output = self.norm(output)
        return output

class Model(TransformerEncoder):
    def __init__(self, num_layers, d_model, nhead, d_ff_factor, dropout, activation, batch_first, norm: bool, 
                num_embeddings, padding_idx, ):
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*d_ff_factor,
            dropout=dropout, activation=activation, batch_first=batch_first,
            norm_first=True
        )
        if norm:
            norm = nn.LayerNorm(d_model ,elementwise_affine=False)
        else:
            norm = None

        super().__init__(layer, num_layers=num_layers, norm=norm)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx)
        self.predictor = nn.Linear(d_model, num_embeddings)

    def forward(self, src: torch.Tensor):
        x = self.embedding(src)
        length = x.shape[0]
        mask = nn.Transformer.generate_square_subsequent_mask(length).to(x.device).to(x.dtype)
        x = super().forward(x, mask)
        return self.predictor(x)
    

