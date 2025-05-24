import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import DyTanh
from utils.mamba_layer import MambaLayer
        

class LatentGate(nn.Module):
    'Maps the latent space to a gate value between 0 and 1, used for gating'
    def __init__(self, latent_dim):
        super(LatentGate, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        nn.init.normal_(self.map[0].weight, mean=0.0, std=0.01)
        
    def forward(self, latent):
        return self.map(latent)

class MambaLatentBiMap(nn.Module):
    'Bi Directional Mamba layer for latent space using the gating mechanism'
    def __init__(self, seq_len, in_dim, d_out, d_state=32, d_conv=4):
        super(MambaLatentBiMap, self).__init__()
        self.mamba = nn.Sequential(
            MambaLayer(in_dim, d_state=d_state, d_out=d_out, d_conv=d_conv),
            MambaLayer(d_out, d_state=d_state, d_out=d_out, d_conv=d_conv),
            DyTanh(seq_len)
        )
        self.reverse_mamba = nn.Sequential(
            MambaLayer(in_dim, d_state=d_state, d_out=d_out, d_conv=d_conv),
            MambaLayer(d_out, d_state=d_state, d_out=d_out, d_conv=d_conv),
            DyTanh(seq_len)
        )
        self.latent_gate = LatentGate(seq_len)
        self.reverse_latent_gate = LatentGate(seq_len)

    def forward(self, x):
        latent_use = self.latent_gate(x)
        reverse_latent_use = self.reverse_latent_gate(x)

        mamba = self.mamba(x) * latent_use
        reverse_x = torch.flip(x, dims=[-1])
        reverse_mamba = self.reverse_mamba(reverse_x) * reverse_latent_use
        reverse_mamba = torch.flip(reverse_mamba, dims=[-1])  # Flip back the reverse_mamba output
        
        return mamba + reverse_mamba

class MambaDecoderLatentBi(nn.Module):
    'Decodes latent space into a format suitable for forecasting'
    def __init__(self, seq_len, in_dim, d_out, d_state=32, d_conv=4):
        super(MambaDecoderLatentBi, self).__init__()

        self.mamba_latent = nn.Sequential(
            MambaLatentBiMap(seq_len, in_dim, d_out,  d_state, d_conv),
            MambaLatentBiMap(seq_len, in_dim, d_out,  d_state, d_conv),
        )

    def forward(self, x):
        x = self.mamba_latent(x)
        return x

