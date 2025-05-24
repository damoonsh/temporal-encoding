import torch
import torch.nn as nn
import torch.nn.functional as F

# class FlatMambaLayer(nn.Module):
#     def __init__(self, d_state=16, d_conv=4, expand=2):
#         super(FlatMambaLayer, self).__init__()
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand

#         # Projections
#         self.in_proj = nn.Linear(1, expand)
#         self.out_proj = nn.Linear(expand, 1)
        
#         # Normalization
#         self.norm = nn.LayerNorm(expand)
        
#         # Depthwise conv
#         self.conv1d = nn.Conv1d(
#             in_channels=expand,
#             out_channels=expand,
#             kernel_size=d_conv,
#             padding=d_conv - 1,
#             groups=expand,
#             bias=False,
#         )

#         # Selective SSM parameters
#         self.A = nn.Parameter(torch.randn(expand, d_state))
#         self.B_proj = nn.Linear(expand, d_state)
#         self.C_proj = nn.Linear(expand, d_state)
#         self.D = nn.Parameter(torch.ones(expand))
#         self.dt_proj = nn.Linear(expand, 1)

#         # Initialize linear layers
#         nn.init.uniform_(self.in_proj.weight, -0.001, 0.001)
#         nn.init.uniform_(self.out_proj.weight, -0.001, 0.001)
#         nn.init.uniform_(self.B_proj.weight, -0.001, 0.001)
#         nn.init.uniform_(self.C_proj.weight, -0.001, 0.001)
#         nn.init.uniform_(self.dt_proj.weight, -0.001, 0.001)

#         # Initialize conv layer
#         nn.init.uniform_(self.conv1d.weight, -0.001, 0.001)

#         # Initialize biases (if any) to zero
#         if self.in_proj.bias is not None:
#             nn.init.zeros_(self.in_proj.bias)
#         if self.out_proj.bias is not None:
#             nn.init.zeros_(self.out_proj.bias)
#         if self.B_proj.bias is not None:
#             nn.init.zeros_(self.B_proj.bias)
#         if self.C_proj.bias is not None:
#             nn.init.zeros_(self.C_proj.bias)
#         if self.dt_proj.bias is not None:
#             nn.init.zeros_(self.dt_proj.bias)

#     def selective_ssm(self, x_conv, B, C, dt):
#         # Discretize A (input-dependent Δ)
#         A_discrete = torch.exp(self.A.unsqueeze(0) * dt.unsqueeze(-1))  # (bs, seq_len, expand, d_state)
        
#         # Parallel scan via cumprod/cumsum (simplified)
#         A_cum = torch.cumprod(A_discrete, dim=1)
#         Bx = x_conv.unsqueeze(-1) * B.unsqueeze(2)  # (bs, seq_len, expand, d_state)
#         h = torch.cumsum(A_cum * Bx, dim=1)
#         y = (h @ C.unsqueeze(-1)).squeeze(-1) + self.D * x_conv
        
#         return y

#     def forward(self, x):
#         # Input: (bs, seq_len) -> (bs, seq_len, 1)
#         x = x.unsqueeze(-1)
        
#         # Project + normalize
#         x = self.in_proj(x)
#         x = self.norm(x)
        
#         # Depthwise conv (causal)
#         x_conv = x.transpose(1, 2)
#         x_conv = self.conv1d(x_conv)[:, :, :x.size(1)]
#         x_conv = x_conv.transpose(1, 2)
        
#         # Selective projections
#         B = self.B_proj(x_conv)  # (bs, seq_len, d_state)
#         C = self.C_proj(x_conv)  # (bs, seq_len, d_state)
#         dt = F.softplus(self.dt_proj(x_conv))  # (bs, seq_len, 1)
        
#         # Selective SSM
#         y = self.selective_ssm(x_conv, B, C, dt)
        
#         # Project back
#         y = self.out_proj(y).squeeze(-1)
#         return y
        
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
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=1)
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=1)
        nn.init.normal_(self.x_proj.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1)

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