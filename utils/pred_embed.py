import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import DyTanh

class PredBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PredBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, in_dim),
            nn.Linear(in_dim, out_dim)
        )
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.001)
                nn.init.zeros_(layer.bias)

    def forward(self, x): return self.block(x)


class PredEmbed(nn.Module):
    def __init__(self, encoder, seq_len, embed_dim):
        super(PredEmbed, self).__init__()
        dim1 = seq_len * embed_dim
        dim2, dim3 = dim1 // 2, dim1 // 4
        self.embed_encoder = encoder
        self.embed_proj = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            DyTanh(dim2),
            nn.Linear(dim2, dim3),
            nn.ReLU(),
            DyTanh(dim3)
        )
        self.raw_proj = nn.Sequential(nn.Linear(seq_len, dim3), nn.ReLU())
        self.pred_comb = nn.Sequential(
            nn.Linear(2 * dim3, 1),
            nn.ReLU()
        )

        nn.init.normal_(self.embed_proj[0].weight, mean=0.0, std=0.001)
        nn.init.normal_(self.embed_proj[3].weight, mean=0.0, std=0.001)
        
        nn.init.normal_(self.raw_proj[0].weight, mean=0.0, std=0.001)
        nn.init.normal_(self.pred_comb[0].weight, mean=0.0, std=0.001)

    def forward(self, X):
        with torch.no_grad(): embed = self.embed_encoder(X) # (bs, seq_len, embed_dim)
        embed = torch.flatten(embed, start_dim=1)
        embed_projection = self.embed_proj(embed)
        X = X.squeeze(-1)
        raw_projection = self.raw_proj(X)

        X_comb = torch.cat([raw_projection, embed_projection], dim=-1)
        pred = self.pred_comb(X_comb)
        return pred
    
class MOEPred(nn.Module):
    def __init__(self, input_shape, num_experts, top_k, expert_layer, load_balance_coef=0.01):
        super(MOEPred, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([expert_layer for _ in range(num_experts)])
        
        self.gate = nn.Sequential(
            nn.Linear(input_shape, num_experts),
            nn.Softmax(dim=-1)
        )

        nn.init.normal_(self.gate[0].weight, mean=0.0, std=0.001)

        self.expert_usage = torch.zeros(num_experts)
        self.target_usage = 1.0 / self.num_experts

        self.load_balance_coef = load_balance_coef

    def load_balance_loss(self):
        """ Load balance loss based on mse loss based on average usage of each expert """
        avg_usage_fraction = self.expert_usage / self.expert_usage.sum()
        
        return torch.mean((avg_usage_fraction - self.target_usage)**2) * self.num_experts * self.load_balance_coef

    def forward(self, X):
        batch_size, _, _ = X.shape
        gate_scores = self.gate(X).topk(self.top_k, dim=-1)[1]
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1) 

        with torch.no_grad():
            expert_mask = torch.zeros_like(gate_scores).scatter_(1, top_k_indices, 1)
            usage_update = expert_mask.sum(dim=0)
            self.expert_usage += usage_update

        weights = F.softmax(top_k_scores, dim=-1) 

        final_output = torch.zeros(batch_size, 1, device=X.device, dtype=X.dtype) 
        flat_top_k_indices = top_k_indices.flatten() 
        flat_weights = weights.flatten()       
        batch_idx_repeated = torch.arange(batch_size, device=X.device).repeat_interleave(self.top_k) 

        for i in range(self.num_experts):
            expert_mask = (flat_top_k_indices == i)
            
            selected_batch_indices = batch_idx_repeated[expert_mask]

            if selected_batch_indices.numel() > 0: 
                expert_input = X[selected_batch_indices] 
                expert_weights = flat_weights[expert_mask].unsqueeze(-1) 
                expert_output = self.experts[i](expert_input) 
                weighted_expert_output = expert_output * expert_weights
                final_output.index_add_(0, selected_batch_indices, weighted_expert_output)

        return final_output