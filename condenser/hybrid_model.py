"""
Hybrid: V3 GenoTransformer layers (compression) + Standard Transformer layers (reasoning).
Top 1-2 layers use standard nn.Linear for full-rank expressiveness.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint
from condenser.genomatrix import GenoTransformerLayer, _R_POOL, generate_R_batch
from condenser.baseline_transformer import BaselineTransformerLayer


class HybridLayer(nn.Module):
    """Either V3 or Standard Transformer layer, switchable."""
    def __init__(self, hidden_dim, num_heads=8, rank=16, seed=0, layer_id=0,
                 alpha=1.0, mode='v3'):
        super().__init__()
        self.mode = mode
        if mode == 'v3':
            self.layer = GenoTransformerLayer(hidden_dim, num_heads, rank, seed, layer_id, alpha)
        else:
            self.layer = BaselineTransformerLayer(hidden_dim, num_heads)

    def forward(self, x):
        return self.layer(x)


class HybridTransformer(nn.Module):
    """First N layers V3, last M layers Standard."""
    def __init__(self, vocab_size=50000, hidden_dim=4096, num_layers=12,
                 num_heads=8, rank=16, alpha=1.0, num_standard=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mode = 'standard' if i >= num_layers - num_standard else 'v3'
            self.layers.append(
                HybridLayer(hidden_dim, num_heads, rank, seed=42, layer_id=i,
                            alpha=alpha, mode=mode)
            )

        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers:
            if layer.mode == 'v3':
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=True)
            else:
                x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def count_trained(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def effective_params(self):
        d = self.hidden_dim
        return self.num_layers * 12 * d * d
