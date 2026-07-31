"""
SCM (Sparse Coding Machine) v3
Completely no-FFN, no-V, no-O, no-multihead architecture.
Content-dependent position routing via extreme low-rank (rank=1).
- Q: D->1, K: D->1 (tiny, no V, no O, no FFN)
- Low-rank transform per layer
- Top-k sparsification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.scale = math.sqrt(rank)

    def forward(self, x):
        return F.linear(x, (self.U @ self.V.T) / self.scale)


class SCM(nn.Module):
    def __init__(self, vocab_size, code_dim=128, rank=8, num_layers=4, k=64, T_max=512):
        super().__init__()
        self.code_dim = code_dim
        self.num_layers = num_layers
        self.k = k

        self.embed = nn.Embedding(vocab_size, code_dim)
        self.pos_embed = nn.Embedding(T_max, code_dim)

        for i in range(num_layers):
            self.add_module(f'transform_{i}', LowRankLinear(code_dim, code_dim, rank))

        self.router_q = nn.Parameter(torch.randn(num_layers, code_dim, 2) * 0.01)
        self.router_k = nn.Parameter(torch.randn(num_layers, code_dim, 2) * 0.01)

        self.norm = nn.LayerNorm(code_dim)
        self.head = nn.Linear(code_dim, vocab_size)

    def sparsify(self, x):
        vals, idx = torch.topk(x, min(self.k, x.shape[-1]), dim=-1)
        sparse = torch.zeros_like(x)
        sparse.scatter_(-1, idx, vals)
        return sparse

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed(torch.arange(T, device=input_ids.device))
        x = self.sparsify(F.relu(x))

        for i in range(self.num_layers):
            q = x @ self.router_q[i]
            k = x @ self.router_k[i]
            attn = torch.einsum('bti,bsi->bts', q, k) / math.sqrt(2)
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            attn = F.softmax(attn + mask, dim=-1)
            context = attn @ x
            x = x + context

            transform = getattr(self, f'transform_{i}')
            x = x + F.relu(transform(x))
            x = self.sparsify(x)

        x = self.norm(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
