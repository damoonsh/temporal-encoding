import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


class JEPA(nn.Module):
    def __init__(self, encoder, predictor, mask_ratio=0.7):
        super().__init__()
        self.encoder = encoder
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False  # EMA encoder is not trained by gradient

        self.predictor = predictor
        self.mask_token = nn.Parameter(torch.randn(1))  # learnable mask token
        self.mask_ratio = mask_ratio

    @torch.no_grad()
    def update_ema(self, momentum=0.998):
        # EMA update of encoder weights
        for ema_param, param in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_param.data = momentum * ema_param.data + (1 - momentum) * param.data

    def random_masking(self, x):
        # x: (batch, seq_len, feats)
        batch, seq_len, feats = x.shape
        mask_ratio = 0.7  # Fixed to mask 70% of indices
        len_mask = int(seq_len * mask_ratio)  # Number of indices to mask
    
        # Randomly select and sort indices to mask (preserve time series order)
        mask_indices = torch.randperm(seq_len, device=x.device)[:len_mask].sort()[0]
    
        # Create binary mask (1 for masked, 0 for kept)
        mask = torch.zeros(batch, seq_len, device=x.device)
        mask[:, mask_indices] = 1
    
        # Apply masking
        x_masked = x.clone()
        mask = mask.unsqueeze(-1).expand(-1, -1, feats)
        x_masked[mask.bool()] = self.mask_token.view(1, 1, 1).expand(batch, seq_len, feats)[mask.bool()]
    
        # Compute ids_keep and ids_restore for compatibility
        # ids_keep = torch.tensor([i for i in range(seq_len) if i not in mask_indices], device=x.device).unsqueeze(0).expand(batch, -1)
        # ids_restore = torch.argsort(torch.cat([ids_keep, mask_indices.unsqueeze(0).expand(batch, -1)], dim=1), dim=1)
    
        return x_masked, mask#, ids_keep, ids_restore

    def forward(self, x):
        x_masked, mask = self.random_masking(x)

        encoded = self.encoder(x_masked)
        mask = mask.expand(-1,-1,encoded.shape[-1])
        
        with torch.no_grad():
            target_encoded = self.ema_encoder(x)
            target_encoded = target_encoded.view((target_encoded.shape[0],-1))

        encoded = encoded.view((x.shape[0],-1))
        mask = mask.contiguous().view((mask.shape[0],-1))
        
        pred = self.predictor(encoded)
        
        loss = F.l1_loss(pred[mask.bool()], target_encoded[mask.bool()])

        return loss, pred, target_encoded, mask