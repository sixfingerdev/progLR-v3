"""
GenoMatrix V5 — Nonlinear Side Paths.
V3 base (R + LoRA) + tiny learnable MLP per operation for nonlinear reasoning.

Each op: out = x @ (R + scaling*U@V^T)^T + tanh(x @ W1) @ W2
Main path: V3 (statistical, free). Side path: MLP (reasoning, tiny).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint


_global_gen = None
_R_POOL = {}

def _get_gen(device):
    global _global_gen
    if _global_gen is None or _global_gen.device != device:
        _global_gen = torch.Generator(device=device)
    return _global_gen


def generate_R_batch(seed, D, device, dtype):
    total = 12 * D * D
    gen = _get_gen(device)
    gen.manual_seed(abs(seed) % (2**31 - 1))
    all_R = torch.randn(total, generator=gen, device=device, dtype=dtype)
    scale = math.sqrt(D)
    off = 0; s = D * D
    R_Q = all_R[off:off+s].view(D, D).div_(scale); off += s
    R_K = all_R[off:off+s].view(D, D).div_(scale); off += s
    R_V = all_R[off:off+s].view(D, D).div_(scale); off += s
    R_O = all_R[off:off+s].view(D, D).div_(scale); off += s
    s = 4 * D * D
    R_FF1 = all_R[off:off+s].view(4*D, D).div_(scale); off += s
    R_FF2 = all_R[off:off+s].view(D, 4*D).div_(scale)
    return {'q': R_Q, 'k': R_K, 'v': R_V, 'o': R_O, 'ff1': R_FF1, 'ff2': R_FF2}


class SideMLP(nn.Module):
    """Tiny learnable MLP for nonlinear side path.
    out = tanh(x @ W1^T + b1) @ W2^T + b2
    W1: (hidden_dim, in_dim), W2: (out_dim, hidden_dim)
    """
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(out_dim, hidden_dim) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        h = F.linear(x, self.W1, self.b1)
        h = torch.tanh(h)
        return F.linear(h, self.W2, self.b2)


class V5LinearFunction(torch.autograd.Function):
    """V3 + Side MLP combined."""
    @staticmethod
    def forward(ctx, x, R, U_lora, V_lora, alpha, rank, W1, b1, W2, b2):
        U_lora = U_lora.to(x.dtype)
        V_lora = V_lora.to(x.dtype)
        R = R.to(x.dtype)
        W1 = W1.to(x.dtype); b1 = b1.to(x.dtype)
        W2 = W2.to(x.dtype); b2 = b2.to(x.dtype)

        scaling = alpha / rank
        LoRA = scaling * (U_lora @ V_lora.transpose(-2, -1))
        W = R + LoRA
        out_main = F.linear(x, W)

        h = F.linear(x, W1, b1).tanh()
        out_side = F.linear(h, W2, b2)

        out = out_main + out_side
        ctx.save_for_backward(x, R, U_lora, V_lora, W1, b1, W2, b2)
        ctx.scaling = scaling
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, R, U_lora, V_lora, W1, b1, W2, b2 = ctx.saved_tensors
        dt = grad_output.dtype
        x = x.to(dt); R = R.to(dt); U_lora = U_lora.to(dt); V_lora = V_lora.to(dt)
        W1 = W1.to(dt); b1 = b1.to(dt); W2 = W2.to(dt); b2 = b2.to(dt)

        in_f = x.shape[-1]; out_f = grad_output.shape[-1]
        x_flat = x.reshape(-1, in_f)
        go_flat = grad_output.reshape(-1, out_f)

        # Side path backward
        h_flat = F.linear(x_flat, W1, b1).tanh()
        go_side = go_flat.clone()
        grad_W2 = go_side.T @ h_flat  # (out_dim, hidden_dim) ← (B*T, out_dim)^T @ (B*T, hidden_dim)
        grad_b2 = go_side.sum(0)
        grad_h = go_side @ W2  # (B*T, hidden_dim) ← (B*T, out_dim) @ (hidden_dim, out_dim)
        grad_h = grad_h * (1 - h_flat * h_flat)  # tanh derivative
        grad_W1 = grad_h.T @ x_flat  # (hidden_dim, in_dim) ← (B*T, hidden_dim)^T @ (B*T, in_dim)
        grad_b1 = grad_h.sum(0)

        # Main path backward (same as V3)
        scaling = ctx.scaling
        grad_W_main = go_flat.T @ x_flat
        grad_U = scaling * (grad_W_main @ V_lora)
        grad_V = scaling * (grad_W_main.T @ U_lora)

        # Gradient to x: main + side
        LoRA = scaling * (U_lora @ V_lora.transpose(-2, -1))
        W = R + LoRA
        grad_x_main = F.linear(grad_output, W.t())
        grad_x_side = grad_h @ W1
        grad_x = grad_x_main + grad_x_side.reshape_as(x)

        return grad_x, None, grad_U, grad_V, None, None, grad_W1, grad_b1, grad_W2, grad_b2


class GenoAttentionV5(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, rank=16, alpha=1.0, side_k=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.rank = rank
        self.alpha = alpha
        D = hidden_dim

        self.U_Q = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_Q = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.U_K = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_K = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.U_V = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_V = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.U_O = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_O = nn.Parameter(torch.randn(D, rank) * 0.01)

        self.side_Q = SideMLP(D, D, side_k)
        self.side_K = SideMLP(D, D, side_k)
        self.side_V = SideMLP(D, D, side_k)
        self.side_O = SideMLP(D, D, side_k)

    def forward(self, x, Rs):
        B, T, D = x.shape
        apply = lambda x_, R, U, V, side: V5LinearFunction.apply(
            x_, R, U, V, self.alpha, self.rank,
            side.W1, side.b1, side.W2, side.b2)

        Q = apply(x, Rs['q'], self.U_Q, self.V_Q, self.side_Q)
        K = apply(x, Rs['k'], self.U_K, self.V_K, self.side_K)
        V = apply(x, Rs['v'], self.U_V, self.V_V, self.side_V)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        out = apply(out, Rs['o'], self.U_O, self.V_O, self.side_O)
        return out


class GenoFFNV5(nn.Module):
    def __init__(self, hidden_dim, ff_mult=4, rank=16, alpha=1.0, side_k=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        D = hidden_dim; H = hidden_dim * ff_mult

        self.U_FF1 = nn.Parameter(torch.randn(H, rank) * 0.01)
        self.V_FF1 = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.U_FF2 = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_FF2 = nn.Parameter(torch.randn(H, rank) * 0.01)

        self.side_FF1 = SideMLP(D, H, side_k)
        self.side_FF2 = SideMLP(H, D, side_k)

    def forward(self, x, Rs):
        apply = lambda x_, R, U, V, side: V5LinearFunction.apply(
            x_, R, U, V, self.alpha, self.rank,
            side.W1, side.b1, side.W2, side.b2)

        h = apply(x, Rs['ff1'], self.U_FF1, self.V_FF1, self.side_FF1)
        h = F.relu(h)
        out = apply(h, Rs['ff2'], self.U_FF2, self.V_FF2, self.side_FF2)
        return out


class GenoTransformerLayerV5(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, rank=16, alpha=1.0, side_k=16,
                 seed=42, layer_id=0):
        super().__init__()
        self.seed = seed
        self.layer_id = layer_id
        self.attn = GenoAttentionV5(hidden_dim, num_heads, rank, alpha, side_k)
        self.ffn = GenoFFNV5(hidden_dim, 4, rank, alpha, side_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        D = x.shape[-1]
        key = (self.seed + self.layer_id * 6, D)
        if key not in _R_POOL or _R_POOL[key]['q'].device != x.device or _R_POOL[key]['q'].dtype != x.dtype:
            _R_POOL[key] = generate_R_batch(*key, x.device, x.dtype)
        Rs = _R_POOL[key]
        x = x + self.attn(self.norm1(x), Rs)
        x = x + self.ffn(self.norm2(x), Rs)
        return x


class GenoTransformerV5(nn.Module):
    def __init__(self, vocab_size=50000, hidden_dim=4096, num_layers=12,
                 num_heads=8, rank=16, alpha=1.0, side_k=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)

        self.layers = nn.ModuleList([
            GenoTransformerLayerV5(hidden_dim, num_heads, rank, alpha, side_k, seed=42, layer_id=i)
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

    def count_trained(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def effective_params(self):
        d = self.hidden_dim
        return self.num_layers * 12 * d * d
