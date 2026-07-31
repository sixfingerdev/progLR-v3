"""
V6: V3 compression on V/O/FFN, standard learned Q/K for OOD preservation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
from condenser.genomatrix import EphemeralLinearFunction, generate_R_batch, _R_POOL


class V6Attention(nn.Module):
    """Q, K = standard learned. V, O = V3 (R + LoRA)."""
    def __init__(self, dim, heads, rank, seed, layer_id, alpha):
        super().__init__()
        self.heads = heads; self.dim = dim; self.head_dim = dim // heads
        self.rank = rank; self.alpha = alpha; self.seed = seed; self.layer_id = layer_id
        
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        
        for name in ['V', 'O']:
            setattr(self, f'U_{name}', nn.Parameter(torch.randn(dim, rank) * 0.01))
            setattr(self, f'V_{name}', nn.Parameter(torch.randn(dim, rank) * 0.01))

    def _get_Rs(self, D, device, dtype):
        key = (self.seed + self.layer_id * 6, D)
        if key not in _R_POOL or _R_POOL[key]['q'].device != device or _R_POOL[key]['q'].dtype != dtype:
            _R_POOL[key] = generate_R_batch(*key, device, dtype)
        return _R_POOL[key]

    def forward(self, x):
        B, T, D = x.shape; H = self.heads; Dh = self.head_dim
        
        Q = self.W_Q(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.W_K(x).view(B, T, H, Dh).transpose(1, 2)
        
        Rs = self._get_Rs(D, x.device, x.dtype)
        R_V, R_O = Rs['v'], Rs['o']
        
        V = EphemeralLinearFunction.apply(x, R_V, self.U_V, self.V_V, self.alpha, self.rank)
        V = V.view(B, T, H, Dh).transpose(1, 2)
        
        attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(Dh), dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        
        return EphemeralLinearFunction.apply(out, R_O, self.U_O, self.V_O, self.alpha, self.rank)


class V6FFN(nn.Module):
    """FFN with V3 compression."""
    def __init__(self, dim, expansion=4, rank=8, seed=0, layer_id=0, alpha=1.0):
        super().__init__()
        self.rank = rank; self.alpha = alpha; self.seed = seed; self.layer_id = layer_id
        for name in ['ffn_1', 'ffn_2']:
            out_d = dim * expansion if name == 'ffn_1' else dim
            in_d = dim if name == 'ffn_1' else dim * expansion
            setattr(self, f'U_{name}', nn.Parameter(torch.randn(out_d, rank) * 0.01))
            setattr(self, f'V_{name}', nn.Parameter(torch.randn(in_d, rank) * 0.01))

    def _get_Rs(self, D, device, dtype):
        key = (self.seed + self.layer_id * 6 + 4, D)
        if key not in _R_POOL or _R_POOL[key]['q'].device != device or _R_POOL[key]['q'].dtype != dtype:
            _R_POOL[key] = generate_R_batch(*key, device, dtype)
        return _R_POOL[key]

    def forward(self, x):
        D = x.shape[-1]
        Rs = self._get_Rs(D, x.device, x.dtype)
        
        hidden = F.relu(EphemeralLinearFunction.apply(x, Rs['ff1'], self.U_ffn_1, self.V_ffn_1, self.alpha, self.rank))
        return EphemeralLinearFunction.apply(hidden, Rs['ff2'], self.U_ffn_2, self.V_ffn_2, self.alpha, self.rank)


class V6Layer(nn.Module):
    def __init__(self, dim, heads, rank, seed, layer_id, alpha):
        super().__init__()
        self.attn = V6Attention(dim, heads, rank, seed, layer_id, alpha)
        self.ffn = V6FFN(dim, 4, rank, seed, layer_id, alpha)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class V6Transformer(nn.Module):
    """V6: Standard Q/K + V3 V/O/FFN."""
    def __init__(self, vocab_size=50000, hidden_dim=4096, num_layers=12,
                 num_heads=8, rank=8, alpha=64, seed=42):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([
            V6Layer(hidden_dim, num_heads, rank, seed, i, alpha)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=True)
        x = self.norm(x)
        return self.lm_head(x)
