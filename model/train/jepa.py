import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class BaseJepaPredictor(nn.Module):
    def __init__(self, d_model=12, d_state=64, d_conv=4):
        super().__init__()
        self.decoder = nn.Sequential(
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
            nn.LayerNorm(d_model),
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
        )

    def forward(self, X):
        return self.decoder(X)

class JEPA(nn.Module):
    def __init__(self, data_encoder, encoder, predictor, rev=False, embed_dim=8, mask_ratio=0.7):
        super().__init__()
        self.encoder = encoder
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False

        self.predictor = predictor
        self.data_encoder = data_encoder
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        self.rev = rev
        if self.rev:
            self.encoder_rev = copy.deepcopy(self.encoder)
            self.ema_encoder_rev = copy.deepcopy(self.encoder_rev)
            for p in self.ema_encoder_rev.parameters():
                p.requires_grad = False
            self.predictor_rev = copy.deepcopy(self.predictor)

    @torch.no_grad()
    def update_ema(self, momentum=0.998):
        for ema_param, param in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_param.data = momentum * ema_param.data + (1 - momentum) * param.data

        if self.rev:
            for ema_param_rev, param_rev in zip(self.ema_encoder_rev.parameters(), self.encoder_rev.parameters()):
                ema_param_rev.data = momentum * ema_param_rev.data + (1 - momentum) * param_rev.data

    def temporal_probabilistic_masking(self, x):
        batch_size, ed, sl = x.shape
        num_masked = int(self.mask_ratio * ed)
    
        # Generate a different random permutation for each batch
        idx = torch.stack([torch.randperm(ed, device=x.device) for _ in range(batch_size)])  # (batch_size, ed)
        masked_idx = idx[:, :num_masked]      # (batch_size, num_masked)
        unmasked_idx = idx[:, num_masked:]    # (batch_size, ed - num_masked)
    
        # Gather masked and unmasked portions for each batch
        x_masked = torch.stack([
            x[i, masked_idx[i], :] for i in range(batch_size)
        ], dim=0)  # (batch_size, num_masked, sl)
    
        x_unmasked = torch.stack([
            x[i, unmasked_idx[i], :] for i in range(batch_size)
        ], dim=0)  # (batch_size, ed - num_masked, sl)
    
        # Create mask for loss computation
        mask = torch.zeros(batch_size, ed, sl, device=x.device)
        for i in range(batch_size):
            mask[i, masked_idx[i], :] = 1
    
        return x_masked, x_unmasked, mask

        
    def forward(self, x):
        x_enc = self.data_encoder(x)  # (bs, ed, sl)
        x_masked, x_unmasked, mask = self.temporal_probabilistic_masking(x_enc)
        # mask: (bs, ed, sl) or (bs, ed) depending on your implementation
    
        # 1. Pass **masked** portion to trainable encoder (context encoder)
        encoded = self.encoder(x_masked)  # x_masked: (bs, ~0.7*ed, sl)
        pred = self.predictor(encoded)
    
        # 2. Pass **full** input to EMA encoder (target encoder)
        with torch.no_grad():
            target_encoded = self.ema_encoder(x_enc)  # x_enc: (bs, ed, sl)
    
        # 3. Select only the masked positions from target_encoded and pred for loss
        mask_bool = mask.bool()  # (bs, ed, sl)
        target_masked = target_encoded[mask_bool] 
        pred_flat = pred.flatten()
        
        elementwise_loss = F.l1_loss(pred_flat, target_masked, reduction='none')
        loss = elementwise_loss.mean()
    
        # Optionally, reverse direction
        if self.rev:
            x_enc_rev = torch.flip(x_enc, dims=[1])
            x_masked_rev, _, mask_rev = self.temporal_probabilistic_masking(x_enc_rev)
    
            encoded_rev = self.encoder_rev(x_masked_rev)
            pred_rev = self.predictor_rev(encoded_rev)
    
            with torch.no_grad():
                target_encoded_rev = self.ema_encoder_rev(x_enc_rev)
    
            pred_masked_rev = pred_rev[mask_rev.bool()]
            target_masked_rev = target_encoded_rev[mask_rev.bool()]
            elementwise_loss_rev = F.l1_loss(pred_masked_rev, target_masked_rev, reduction='none')
            loss_rev = elementwise_loss_rev.mean()
    
            total_loss = loss + loss_rev
            return total_loss
    
        return loss

class JEPAEncoder(nn.Module):
    ''' JEPA pipelines, after JEPA is trained this will be used as the processing bit for forecasting '''
    def __init__(self, jepa_model: JEPA):
        super().__init__()
        self.jepa = jepa_model
        self.jepa.eval() 
        
        self.data_encoder = self.jepa.data_encoder
        self.encoder = self.jepa.encoder
        self.rev = self.jepa.rev
        if self.rev:
            self.encoder_rev = self.jepa.encoder_rev
            for param in self.encoder_rev.parameters():
                param.requires_grad = False

        for param in self.data_encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
            

    @torch.no_grad() 
    def forward(self, x):
        x_enc = self.data_encoder(x)
        latent = self.encoder(x_enc)

        if self.rev:
            x_enc_rev = torch.flip(x_enc, dims=[1])
            latent_rev_raw = self.encoder_rev(x_enc_rev)
            latent_rev_processed = torch.flip(latent_rev_raw, dims=[1])
            latent = latent + latent_rev_processed
            
        return latent