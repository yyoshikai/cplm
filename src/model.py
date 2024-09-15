import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.TransformerEncoder):
    def __init__(self, num_layers, d_model, d_ff_factor, dropout, activation, norm):
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=d_model*d_ff_factor,
            dropout=dropout, activation=activation, 
            norm_first=True
        )
        if norm:
            norm = nn.LayerNorm(d_model ,elementwise_affine=False)
        else:
            norm = None

        super().__init__(layer, num_layers=num_layers, norm=norm)
