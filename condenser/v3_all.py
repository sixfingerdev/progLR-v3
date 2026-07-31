"""
V3-All Transformer:
- V3-Embed: R + a/r*U@V^T for embedding/head (500x compression)
- Progressive Low-Rank: standard + low-rank layers (2-5x compression)
- One unified format: W = R + a/r * U@V^T
  - Embed: R=random (proven)
  - Std layers: R=learned dense (full-rank)
  - LR layers: R=0, pure U@V^T

Architecture engineering for 6GB consumer GPU scaling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class V3Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, rank=8, alpha=64.0):
        super().__init__()
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        R = torch.randn(vocab_size, hidden_dim) / math.sqrt(hidden_dim)
        self.register_buffer('R', R)
        self.U = nn.Parameter(torch.randn(vocab_size, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)

    def forward_embed(self, input_ids):
        return F.embedding(input_ids, self.R) + \
               (self.scale * F.embedding(input_ids, self.U) @ self.V.T)

    def forward_head(self, x):
        return F.linear(x, self.R) + \
               (self.scale * (x @ self.V) @ self.U.T)


class V3Linear(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=64.0, lr_mode='lowrank'):
        super().__init__()
        self.scale = alpha / max(rank, 1)
        self.lr_mode = lr_mode
        if lr_mode == 'standard':
            self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02 / math.sqrt(in_dim))
        elif lr_mode == 'lowrank':
            self.U = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
            self.V = nn.Parameter(torch.randn(in_dim, rank) * 0.02)

    def forward(self, x):
        if self.lr_mode == 'standard':
            return F.linear(x, self.W)
        else:
            return self.scale * (x @ self.V) @ self.U.T


class StdAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_Q = V3Linear(hidden_dim, hidden_dim, 0, lr_mode='standard')
        self.W_K = V3Linear(hidden_dim, hidden_dim, 0, lr_mode='standard')
        self.W_V = V3Linear(hidden_dim, hidden_dim, 0, lr_mode='standard')
        self.W_O = V3Linear(hidden_dim, hidden_dim, 0, lr_mode='standard')

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn = F.softmax(attn + mask, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.W_O(out)


class LRAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, rank, alpha):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_Q = V3Linear(hidden_dim, hidden_dim, rank, alpha, 'lowrank')
        self.W_K = V3Linear(hidden_dim, hidden_dim, rank, alpha, 'lowrank')
        self.W_V = V3Linear(hidden_dim, hidden_dim, rank, alpha, 'lowrank')
        self.W_O = V3Linear(hidden_dim, hidden_dim, rank, alpha, 'lowrank')

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn = F.softmax(attn + mask, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.W_O(out)


class StdFFN(nn.Module):
    def __init__(self, hidden_dim, ff_mult=4):
        super().__init__()
        ff_dim = hidden_dim * ff_mult
        self.W1 = V3Linear(hidden_dim, ff_dim, 0, lr_mode='standard')
        self.W2 = V3Linear(ff_dim, hidden_dim, 0, lr_mode='standard')

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


class LRFFN(nn.Module):
    def __init__(self, hidden_dim, ff_mult, rank, alpha):
        super().__init__()
        ff_dim = hidden_dim * ff_mult
        self.W1 = V3Linear(hidden_dim, ff_dim, rank, alpha, 'lowrank')
        self.W2 = V3Linear(ff_dim, hidden_dim, rank, alpha, 'lowrank')

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


class StdLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = StdAttention(hidden_dim, num_heads)
        self.ffn = StdFFN(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LRLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, rank, alpha):
        super().__init__()
        self.attn = LRAttention(hidden_dim, num_heads, rank, alpha)
        self.ffn = LRFFN(hidden_dim, 4, rank, alpha)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class V3AllTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=12, num_heads=12,
                 embed_rank=8, hidden_rank=32, alpha=64.0,
                 num_std_layers=2, layer_pattern=None):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.v3_embed = V3Embedding(vocab_size, hidden_dim, embed_rank, alpha)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)

        if layer_pattern is None:
            n_std = min(num_std_layers, num_layers)
            layer_pattern = ['std'] * n_std + ['lr'] * (num_layers - n_std)

        layers = []
        for kind in layer_pattern:
            if kind == 'std':
                layers.append(StdLayer(hidden_dim, num_heads))
            else:
                layers.append(LRLayer(hidden_dim, num_heads, hidden_rank, alpha))
        self.layers = nn.ModuleList(layers)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.v3_embed.forward_embed(input_ids) + self.pos_embed[:, :T]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.v3_embed.forward_head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def vram_estimate(self):
        trained = self.count_params()
        model_mb = trained * 2 / 1e6
        optim_mb = trained * 8 / 1e6
        act_mb = self.hidden_dim * 512 * 4 * 2 / 1e6
        total_gb = (model_mb + optim_mb + act_mb) / 1000
        return {
            'trained_m': trained / 1e6,
            'model_mb': model_mb,
            'optim_mb': optim_mb,
            'act_mb': act_mb,
            'total_gb': total_gb,
            'fits_6gb': total_gb < 6.0,
        }
