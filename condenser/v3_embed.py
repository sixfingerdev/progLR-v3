"""
V3-Embed: V3 weight generation for embedding/head layers.
W = R + alpha/rank * U @ V^T (same as V3 layer compression).
Transformer layers stay standard → reasoning preserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from condenser.baseline_transformer import BaselineTransformerLayer


def build_R(vocab_size, hidden_dim, seed=0, device=None, dtype=torch.float32):
    """Deterministic frozen base: seed-regenerable, no need to store 750MB in ckpt."""
    g = torch.Generator(device='cpu' if device is None else 'cpu')
    g.manual_seed(int(seed))
    R = torch.randn(vocab_size, hidden_dim, generator=g) / math.sqrt(hidden_dim)
    return R.to(device=device or 'cpu', dtype=dtype)


class V3WeightEmbedding(nn.Module):
    """Embedding via V3 weight generation: W = R + alpha/r * U @ V^T."""
    def __init__(self, vocab_size, hidden_dim, rank=8, alpha=1.0, r_seed=0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.r_seed = int(r_seed)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        R = build_R(vocab_size, hidden_dim, seed=self.r_seed)
        self.register_buffer('R', R)
        self.U = nn.Parameter(torch.randn(vocab_size, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)

    def regen_R(self, device=None, dtype=None):
        """Regenerate frozen R from seed (for compact-ckpt load)."""
        R = build_R(self.vocab_size, self.hidden_dim, seed=self.r_seed,
                    device=device or self.R.device, dtype=dtype or self.R.dtype)
        self.R.data = R
        return R

    def state_dict_compact(self):
        """State without the frozen R buffer (~750MB @250K vocab)."""
        sd = self.state_dict()
        sd.pop('R', None)
        return sd

    def load_compact(self, sd, strict=False):
        """Load U/V (+config), then regen R from seed. Accepts full or compact dicts."""
        if 'R' not in sd:
            self.regen_R()
        missing, unexpected = self.load_state_dict(sd, strict=False), None
        return missing

    def forward_embed(self, input_ids):
        R_emb = F.embedding(input_ids, self.R)
        U_emb = F.embedding(input_ids, self.U)
        lora = (self.alpha / self.rank) * (U_emb @ self.V.T)
        return R_emb + lora

    def forward_head(self, x):
        R_logits = F.linear(x, self.R)
        lora = (self.alpha / self.rank) * (x @ self.V) @ self.U.T
        return R_logits + lora


class V3EmbedTransformer(nn.Module):
    """Standard transformer + V3 weight-generated embedding/head."""
    def __init__(self, vocab_size=50000, hidden_dim=768, num_layers=12,
                 num_heads=12, embed_rank=8, alpha=1.0):
        super().__init__()
        self.v3_embed = V3WeightEmbedding(vocab_size, hidden_dim, embed_rank, alpha)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([
            BaselineTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.v3_embed.forward_embed(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.v3_embed.forward_head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
