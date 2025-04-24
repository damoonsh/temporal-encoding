import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DyTanh
        
class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, d_out=128):
        super(MambaLayer, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        # Convolution
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)  # B, C, delta
        self.dt_proj = nn.Linear(1, self.d_inner)
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_out)

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.ones(d_state, d_state) * 0.1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers with normal distribution (mean=0, std=0.001)
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.x_proj.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.1)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape
        # Input projection and split
        x_and_res = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x, res = x_and_res.split(self.d_inner, dim=-1)  # (batch, seq_len, d_inner)
        
        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # (batch, d_inner, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        x = F.silu(x)

        # SSM parameters
        x_ssm = self.x_proj(x)  # (batch, seq_len, 2*d_state + 1)
        B, C, delta = x_ssm.split([self.d_state, self.d_state, 1], dim=-1)  # (batch, seq_len, d_state), (batch, seq_len, 1)
        delta = F.softplus(self.dt_proj(delta))  # (batch, seq_len, d_inner)
        
        # Simplified SSM
        A = -torch.exp(self.A_log)  # (d_state, d_inner)
        y = torch.zeros_like(x)  # (batch, seq_len, d_inner)
        yt = torch.zeros(batch, self.d_state, 1, device=x.device)  # (batch, d_state, 1)
        
        for t in range(seq_len):
            xt = x[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)
            Bt = B[:, t, :].unsqueeze(-1)  # (batch, d_state, 1)
            Ct = C[:, t, :].unsqueeze(-1)  # (batch, d_state, 1)
            
            delta_t = delta[:, t, :].unsqueeze(-1).mean(dim=1, keepdim=True)  # (batch, 1, 1)
            scaled_A = delta_t * A
            yt = torch.exp(scaled_A) @ yt + Bt * xt.mean(dim=1, keepdim=True)  # (batch, d_state, 1)
            yt = yt / (yt.norm(dim=1, keepdim=True) + 1e-6)
            y[:, t, :] = (Ct.transpose(1, 2) @ yt).squeeze(1)  # (batch, d_inner)
        
        # Output projection
        y = y + res  # Residual connection
        y = self.out_proj(y)  # (batch, seq_len, d_model)
        return y

class MambaEncoder(nn.Module):
    def __init__(self, seq_len, in_dim, d_state, d_out, d_conv):
        super().__init__()
        self.mamba = nn.Sequential(
            MambaLayer(in_dim, d_state=d_state, d_out=d_out, d_conv=d_conv),
            DyTanh((seq_len, d_out)),
            MambaLayer(d_out, d_state=d_state, d_out=d_out, d_conv=d_conv),
            DyTanh((seq_len, d_out))
        )

    def forward(self, x):
        return self.mamba(x)
    
class MambaEncoderU(nn.Module):
    def __init__(self, seq_len, in_dim, d_state, d_out):
        super().__init__()
        self.mamba = nn.Sequential(
            MambaLayer(1, d_state=d_state, d_out=4),
            DyTanh((seq_len, 4)),
            MambaLayer(4, d_state=d_state, d_conv=8, expand=2, d_out=8),
            DyTanh((seq_len, 8))
        )

    def forward(self, x):
        return self.mamba(x)