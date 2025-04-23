import torch
import torch.nn as nn
import torch.nn.functional as F

class DyTanh(nn.Module):
    def __init__(self, shape, mean=0.1, std=0.001):
        super(DyTanh, self).__init__()
        self.scale = nn.Parameter(torch.randn(shape) * std + mean)
        self.shift = nn.Parameter(torch.randn(shape) * std + mean)
        self.alpha = nn.Parameter(torch.randn(shape) * std + mean)

    def forward(self, x):
        return self.scale * torch.tanh(self.alpha * x) + self.shift