import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import DyTanh

from utils.mamba_layer import MambaLayer
        
class MambaEncoder(nn.Module):
    def __init__(self, seq_len, in_dim=1, d_out=8, d_state=32, d_conv=4):
        super().__init__()

        dim1, dim2 = in_dim * 2, in_dim * 4

        self.linear_proj = nn.Sequential(
            nn.Linear(in_dim, dim1),
            nn.Linear(dim1, dim2),
            nn.Linear(dim2, d_out),
        )

        for layer in self.linear_proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.001)
                nn.init.zeros_(layer.bias)

        self.mamba = nn.Sequential(
            self.linear_proj,
            DyTanh(d_out),
            MambaLayer(d_out, d_state=d_state, d_out=d_out, d_conv=d_conv),
            MambaLayer(d_out, d_state=d_state, d_out=d_out, d_conv=d_conv),
            DyTanh((seq_len, d_out))
        )

    def forward(self, x):
        return self.mamba(x)