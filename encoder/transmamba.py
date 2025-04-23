import torch
import torch.nn as nn
import math

class PerformerAttention(nn.Module):
    def __init__(self, d_model, num_features=256, kernel_type="relu"):
        super(PerformerAttention, self).__init__()
        self.d_model = d_model
        self.num_features = num_features  # Number of random features for FAVOR+
        self.kernel_type = kernel_type

        # Random feature projection matrices for Q and K
        self.random_features = nn.Parameter(
            torch.randn(d_model, num_features) / math.sqrt(d_model)
        )

    def _compute_kernel(self, x):
        """Apply kernel transformation (e.g., ReLU or exp) to input."""
        projection = x @ self.random_features
        if self.kernel_type == "relu":
            return torch.relu(projection)
        elif self.kernel_type == "exp":
            return torch.exp(projection)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def forward(self, q, k, v):
        """Compute Performer attention using kernel-based approximation.
        
        Args:
            q: Query tensor (batch_size, seq_len, d_model)
            k: Key tensor (batch_size, seq_len, d_model)
            v: Value tensor (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Compute kernelized Q and K
        q_prime = self._compute_kernel(q)  # (batch_size, seq_len, num_features)
        k_prime = self._compute_kernel(k)  # (batch_size, seq_len, num_features)

        # Compute prefix sums for efficient attention
        k_prime_sum = k_prime.sum(dim=1, keepdim=True)  # (batch_size, 1, num_features)
        kv_product = torch.einsum("bsm,bsn->bmn", k_prime, v)  # (batch_size, num_features, d_model)

        # Compute attention output: (Q' * (K'V)) / (Q' * sum(K'))
        numerator = torch.einsum("bsm,bmn->bsn", q_prime, kv_product)  # (batch_size, seq_len, d_model)
        denominator = torch.einsum("bsm,btm->bst", q_prime, k_prime_sum)  # (batch_size, seq_len, 1)
        output = numerator / (denominator + 1e-8)  # Avoid division by zero

        return output

class PerformerLayer(nn.Module):
    def __init__(self, d_model, num_features=256, kernel_type="relu", ffn_dim=512, dropout=0.1):
        super(PerformerLayer, self).__init__()
        self.attention = PerformerAttention(d_model, num_features, kernel_type)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """Forward pass of the Performer layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # LayerNorm and attention
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        # Performer attention
        attn_output = self.attention(q, k, v)
        attn_output = self.out_proj(attn_output)
        x = x + self.dropout(attn_output)  # Residual connection
        
        # Feedforward and second residual connection
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        
        return x