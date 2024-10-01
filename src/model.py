from typing import Union, Callable, Optional, Tuple
import copy
import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.functional import _canonical_mask, _none_or_dtype, _in_projection_packed, _in_projection, pad, softmax, dropout, linear, scaled_dot_product_attention


class Model(nn.TransformerEncoder):
    def __init__(self, num_layers, d_model, nhead, d_ff_factor, dropout, activation, norm: bool, 
                num_embeddings, padding_idx, ):
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*d_ff_factor,
            dropout=dropout, activation=activation,
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
        mask = nn.Transformer.generate_square_subsequent_mask(len(x), 
                device=x.device, dtype=x.dtype)
        x = super().forward(x, mask=mask, is_causal=True)
        return self.predictor(x)


