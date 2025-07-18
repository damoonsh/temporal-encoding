

class DyTanh(nn.Module):
    def __init__(self, shape, mean=0.5, std=0.001):
        super(DyTanh, self).__init__()
        self.scale = nn.Parameter(torch.randn(shape) * std + mean)
        self.shift = nn.Parameter(torch.randn(shape) * std + mean)
        self.alpha = nn.Parameter(torch.randn(shape) * std + mean)

    def forward(self, x):
        return self.scale * torch.tanh(self.alpha * x) + self.shift

class PatchConvBlock(nn.Module):
    def __init__(self, num_patch, patch_len, proj_dim, out_channels, kernel_size, stride):
        super(PatchConvBlock, self).__init__()
        self.linear = nn.Linear(patch_len, proj_dim)
        self.conv1d = nn.Conv1d(num_patch, out_channels, kernel_size, stride)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
            
        nn.init.kaiming_normal_(self.conv1d.weight, nonlinearity='relu')
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)
        
    def forward(self, x):
        # x: (bs, num_patch, patch_len)
        x = self.linear(x) 
        x = self.conv1d(x)
        return x

class TempEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        num_patch_1, patch_len_1, proj_dim_1, out_channels_1, kernel_size_1, stride_1 = 5, 16, 16, 5, 5, 1
        num_patch_2, patch_len_2, proj_dim_2, out_channels_2, kernel_size_2, stride_2 = 5, 12, 16, 10, 5, 1
        
        self.block1 = PatchConvBlock(num_patch_1, patch_len_1, proj_dim_1, out_channels_1, kernel_size_1, stride_1)
        self.norm1 = DyTanh(12)
        
        self.block2 = PatchConvBlock(num_patch_2, patch_len_2, proj_dim_2, out_channels_2, kernel_size_2, stride_2)
        self.norm2 = DyTanh(12)

    def forward(self, x):
        x = self.norm1(self.block1(x))
        x = self.norm2(self.block2(x))
        return x