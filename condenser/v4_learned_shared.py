"""
V4: Learned Shared Random Base + Per-Layer LoRA.

Key differences from V3:
1. R is learned (R_init + U_R@V_R^T) — not purely random
2. R is SHARED across all layers — consistent projections
3. Each layer adds its own LoRA correction
4. Gradient flows through R to update the shared base
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class V4LinearFunction(torch.autograd.Function):
    """Like EphemeralLinearFunction but grad flows through R."""
    @staticmethod
    def forward(ctx, x, R, U, V, alpha, rank):
        U = U.to(x.dtype)
        V = V.to(x.dtype)
        scaling = alpha / rank
        LoRA = scaling * (U @ V.transpose(-2, -1))
        W = R + LoRA
        out = F.linear(x, W)
        ctx.save_for_backward(x, R, U, V)
        ctx.scaling = scaling
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, R, U, V = ctx.saved_tensors
        dt = grad_output.dtype
        R = R.to(dt); U = U.to(dt); V = V.to(dt); x = x.to(dt)
        in_f = x.shape[-1]; out_f = grad_output.shape[-1]
        x_flat = x.reshape(-1, in_f); go_flat = grad_output.reshape(-1, out_f)
        scaling = ctx.scaling
        grad_W = go_flat.T @ x_flat  # (out_f, in_f)
        grad_U = scaling * (grad_W @ V)
        grad_V = scaling * (grad_W.T @ U)
        LoRA = scaling * (U @ V.transpose(-2, -1))
        W = R + LoRA
        grad_x = F.linear(grad_output, W.t())
        return grad_x, grad_W, grad_U, grad_V, None, None


class V4SharedBase(nn.Module):
    """Learned shared base matrix: R = R_init + U_R @ V_R^T."""
    def __init__(self, out_dim, in_dim, shared_rank=16, seed=42):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        R_init = torch.randn(out_dim, in_dim, generator=g) / math.sqrt(in_dim)
        self.register_buffer('R_init', R_init)
        self.U = nn.Parameter(torch.randn(out_dim, shared_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(in_dim, shared_rank) * 0.01)

    def get_R(self):
        return self.R_init + self.U @ self.V.T


class V4Attention(nn.Module):
    def __init__(self, dim, heads, rank, shared_Rs, alpha, seed, layer_id):
        super().__init__()
        self.heads = heads; self.dim = dim; self.head_dim = dim // heads
        self.rank = rank; self.alpha = alpha; self.shared_Rs = shared_Rs
        for name in ['Q', 'K', 'V', 'O']:
            setattr(self, f'U_{name}', nn.Parameter(torch.randn(dim, rank) * 0.01))
            setattr(self, f'V_{name}', nn.Parameter(torch.randn(dim, rank) * 0.01))

    def forward(self, x):
        B, T, D = x.shape; H = self.heads; Dh = self.head_dim

        def apply(name, x_in):
            R = self.shared_Rs[f'attn_{name}'].get_R()
            U = getattr(self, f'U_{name}'); V = getattr(self, f'V_{name}')
            out = V4LinearFunction.apply(x_in, R, U, V, self.alpha, self.rank)
            if name in ['Q', 'K']:
                out = out / math.sqrt(Dh)
            return out

        Q = apply('Q', x).view(B, T, H, Dh).transpose(1, 2)
        K = apply('K', x).view(B, T, H, Dh).transpose(1, 2)
        V = apply('V', x).view(B, T, H, Dh).transpose(1, 2)

        attn = F.softmax(Q @ K.transpose(-2, -1), dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return apply('O', out)


class V4FFN(nn.Module):
    def __init__(self, dim, expansion=4, rank=8, shared_Rs=None, alpha=1.0):
        super().__init__()
        self.rank = rank; self.alpha = alpha; self.shared_Rs = shared_Rs
        for name in ['ffn_1', 'ffn_2']:
            out_d = dim * expansion if name == 'ffn_1' else dim
            in_d = dim if name == 'ffn_1' else dim * expansion
            setattr(self, f'U_{name}', nn.Parameter(torch.randn(out_d, rank) * 0.01))
            setattr(self, f'V_{name}', nn.Parameter(torch.randn(in_d, rank) * 0.01))

    def forward(self, x):
        R1 = self.shared_Rs['ffn_1'].get_R()
        R2 = self.shared_Rs['ffn_2'].get_R()
        hidden = F.relu(V4LinearFunction.apply(x, R1, self.U_ffn_1, self.V_ffn_1, self.alpha, self.rank))
        return V4LinearFunction.apply(hidden, R2, self.U_ffn_2, self.V_ffn_2, self.alpha, self.rank)


class V4Layer(nn.Module):
    def __init__(self, dim, heads, rank, shared_Rs, alpha, seed, layer_id):
        super().__init__()
        self.attn = V4Attention(dim, heads, rank, shared_Rs, alpha, seed, layer_id)
        self.ffn = V4FFN(dim, 4, rank, shared_Rs, alpha)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class V4Transformer(nn.Module):
    """V4: Learned shared random base + per-layer LoRA."""
    def __init__(self, vocab_size=50000, hidden_dim=4096, num_layers=12,
                 num_heads=8, rank=8, shared_rank=16, alpha=64, seed=42):
        super().__init__()
        self.hidden_dim = hidden_dim; self.num_layers = num_layers

        self.shared_Rs = nn.ModuleDict()
        for name, out_d, in_d in [
            ('attn_Q', hidden_dim, hidden_dim), ('attn_K', hidden_dim, hidden_dim),
            ('attn_V', hidden_dim, hidden_dim), ('attn_O', hidden_dim, hidden_dim),
            ('ffn_1', hidden_dim*4, hidden_dim), ('ffn_2', hidden_dim, hidden_dim*4),
        ]:
            s = seed + sum(ord(c) for c in name)
            self.shared_Rs[name] = V4SharedBase(out_d, in_d, shared_rank, s)

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([
            V4Layer(hidden_dim, num_heads, rank, self.shared_Rs, alpha, seed, i)
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

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def standard_params(self):
        d = self.hidden_dim; L = self.num_layers
        v = self.token_embedding.num_embeddings
        return L * (4 * d * d + 2 * 4 * d * d) + v * d
