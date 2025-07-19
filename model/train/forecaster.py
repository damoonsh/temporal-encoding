import torch.nn as nn

class SimpleCNNForecaster(nn.Module):
    def __init__(self, num_patch, embed_dim, patch_len, 
                 conv1_out_channels=128, conv2_out_channels=256, 
                 conv_kernel_size=3, conv_stride=1, linear_hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, 
                               out_channels=conv1_out_channels, 
                               kernel_size=conv_kernel_size, 
                               stride=conv_stride, padding=1)
        self.conv2 = nn.Conv1d(in_channels=conv1_out_channels, 
                               out_channels=conv2_out_channels, 
                               kernel_size=conv_kernel_size, 
                               stride=conv_stride, padding=1)
        
        out_num_patch = num_patch  

        self.flatten_dim = conv2_out_channels * out_num_patch
        self.linear = nn.Linear(self.flatten_dim, patch_len)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1) 
        x = self.linear(x)  
        return x

class FlatFNN(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.flatten = nn.Flatten(start_dim=1)
        self.layers = nn.Sequential(
            nn.Linear(120, 60),
            DyTanh(60),
            nn.Linear(60, 30),
            DyTanh(30),
            nn.Linear(30, 16)
        )
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        latent = self.encoder(x)
        flat_latent = self.flatten(latent)
        forecast = self.layers(flat_latent)
        return forecast