import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaselineAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.W_O(out)


class BaselineFFN(nn.Module):
    def __init__(self, hidden_dim, ff_mult=4):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim * ff_mult, bias=False)
        self.W2 = nn.Linear(hidden_dim * ff_mult, hidden_dim, bias=False)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


class BaselineTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.attn = BaselineAttention(hidden_dim, num_heads)
        self.ffn = BaselineFFN(hidden_dim, 4)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size=50000, hidden_dim=4096, num_layers=12, num_heads=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([
            BaselineTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
