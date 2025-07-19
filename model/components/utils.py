import torch
import torch.nn as nn
import torch.nn.functional as F

import copy 

def plot_losses(loss_dict, title="Training and Validation Loss"):
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(next(iter(loss_dict.values()))) + 1)

    for loss_name, loss_values in loss_dict.items():
        if type(loss_values) == list:
            plt.plot(epochs, loss_values, label=loss_name.replace('_', ' ').title())

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

class DyTanh(nn.Module):
    def __init__(self, shape, mean=0.1, std=0.001):
        super(DyTanh, self).__init__()
        self.scale = nn.Parameter(torch.randn(shape) * std + mean)
        self.shift = nn.Parameter(torch.randn(shape) * std + mean)
        self.alpha = nn.Parameter(torch.randn(shape) * std + mean)

    def forward(self, x):
        return self.scale * torch.tanh(self.alpha * x) + self.shift
    
class DyTanhInstance(nn.Module):
    def __init__(self, embed_dim, mean=0.1, std=0.001):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)
        self.shift = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)
        self.alpha = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)

    def forward(self, x):
        # x: (bs, seq_len, embed_dim)
        return self.scale * torch.tanh(self.alpha * x) + self.shift

class DyTanhBatch(nn.Module):
    def __init__(self, embed_dim, mean=0.1, std=0.001):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)
        self.shift = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)
        self.alpha = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)

    def forward(self, x):
        # x: (bs, seq_len, embed_dim)
        return self.scale * torch.tanh(self.alpha * x) + self.shift

class DyTanhGroup(nn.Module):
    def __init__(self, embed_dim, num_groups=4, mean=0.1, std=0.001):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = embed_dim // num_groups
        assert embed_dim % num_groups == 0, "embed_dim must be divisible by num_groups"
        self.scale = nn.Parameter(torch.randn(1, 1, num_groups, 1) * std + mean)
        self.shift = nn.Parameter(torch.randn(1, 1, num_groups, 1) * std + mean)
        self.alpha = nn.Parameter(torch.randn(1, 1, num_groups, 1) * std + mean)

    def forward(self, x):
        # x: (bs, seq_len, embed_dim)
        # Reshape to (bs, seq_len, num_groups, group_size)
        x_grouped = x.view(x.size(0), x.size(1), self.num_groups, -1)
        # Apply group-wise scaling
        out = self.scale * torch.tanh(self.alpha * x_grouped) + self.shift
        # Reshape back
        return out.view_as(x)

    
class BaseJepaPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        dim1, dim2 = input_size, input_size // 2
        self.net = nn.Sequential( 
            nn.Linear(dim1, dim2),
            DyTanh(dim2),
            nn.Linear(dim2, dim1)
        )
        nn.init.normal_(self.net[0].weight, mean=0.0, std=0.001)
        nn.init.normal_(self.net[2].weight, mean=0.0, std=0.001)

    def forward(self, x):
        return self.net(x)
    