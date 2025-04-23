import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DyTanh

class PredEmbed(nn.Module):
    def __init__(self, encoder):
        super(PredEmbed, self).__init__()
        self.embed_encoder = encoder
        self.embed_proj = nn.Sequential(
            nn.Linear(8000, 2048),
            nn.ReLU(),
            DyTanh(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            DyTanh(512)
        )
        self.raw_proj = nn.Sequential(nn.Linear(1000, 512), nn.ReLU())
        self.pred_comb = nn.Sequential(
            nn.Linear(1024, 1),
            nn.ReLU()
        )

        nn.init.normal_(self.embed_proj[0].weight, mean=0.0, std=0.001)
        nn.init.normal_(self.embed_proj[3].weight, mean=0.0, std=0.001)
        
        nn.init.normal_(self.raw_proj[0].weight, mean=0.0, std=0.001)
        nn.init.normal_(self.pred_comb[0].weight, mean=0.0, std=0.001)

    def forward(self, X):
        with torch.no_grad(): embed = self.embed_encoder(X)
        embed = torch.flatten(embed, start_dim=1)
        embed_projection = self.embed_proj(embed)
        X = X.squeeze(-1)
        raw_projection = self.raw_proj(X)

        X_comb = torch.cat([raw_projection, embed_projection], dim=-1)
        pred = self.pred_comb(X_comb)
        return pred