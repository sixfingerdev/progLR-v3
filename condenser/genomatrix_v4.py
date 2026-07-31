"""
GenoMatrix V4 — Spectral Steering.
R = U @ diag(exp(s)) @ V^T  (learnable singular values on fixed random orthogonal basis)
W = R + (alpha/rank) * U_lora @ V_lora^T

U, V from QR of randn — fixed. s = learnable log singular values — D params per op.
LoRA provides rank-r precision corrections on top.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint


_global_gen = None

def _get_gen(device):
    global _global_gen
    if _global_gen is None or _global_gen.device != device:
        _global_gen = torch.Generator(device=device)
    return _global_gen


def _make_orthogonal_pair(D, device, dtype=torch.float32):
    """Generate two fixed orthogonal matrices U, V from QR."""
    gen = _get_gen(device)
    M = torch.randn(D, D, generator=gen, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(M)
    gen.manual_seed(abs(hash((D, 1))) % (2**31 - 1))
    M2 = torch.randn(D, D, generator=gen, device=device, dtype=dtype)
    Q2, _ = torch.linalg.qr(M2)
    return Q.contiguous(), Q2.contiguous()


class SpectralR(nn.Module):
    """R = U @ diag(exp(s)) @ V^T with learnable s, fixed U/V."""
    def __init__(self, out_d, in_d, seed_hint=0, device=None, dtype=torch.float32):
        super().__init__()
        self.out_d = out_d
        self.in_d = in_d
        k = min(out_d, in_d)

        # Fixed orthogonal matrices from QR
        gen = _get_gen(device if device else torch.device('cpu'))
        gen.manual_seed(abs(seed_hint + hash((out_d, in_d, 0))) % (2**31 - 1))
        M = torch.randn(out_d, k, generator=gen, device=device, dtype=dtype)
        U, _ = torch.linalg.qr(M)
        gen.manual_seed(abs(seed_hint + hash((out_d, in_d, 1))) % (2**31 - 1))
        M = torch.randn(in_d, k, generator=gen, device=device, dtype=dtype)
        V, _ = torch.linalg.qr(M)

        self.register_buffer('U_fixed', U.contiguous())
        self.register_buffer('V_fixed', V.contiguous())
        self.log_s = nn.Parameter(torch.zeros(k))

    def forward(self):
        s = torch.exp(self.log_s)
        return (self.U_fixed * s) @ self.V_fixed.T


class SpectralLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, R, U_lora, V_lora, alpha, rank):
        U_lora = U_lora.to(x.dtype)
        V_lora = V_lora.to(x.dtype)
        R = R.to(x.dtype)
        scaling = alpha / rank
        LoRA = scaling * (U_lora @ V_lora.transpose(-2, -1))
        W = R + LoRA
        out = F.linear(x, W)
        ctx.save_for_backward(x, R, U_lora, V_lora)
        ctx.scaling = scaling
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, R, U_lora, V_lora = ctx.saved_tensors
        dt = grad_output.dtype
        R = R.to(dt); U_lora = U_lora.to(dt); V_lora = V_lora.to(dt)
        x = x.to(dt)
        in_f = x.shape[-1]; out_f = grad_output.shape[-1]
        x_flat = x.reshape(-1, in_f); go_flat = grad_output.reshape(-1, out_f)
        scaling = ctx.scaling
        grad_W = go_flat.T @ x_flat
        grad_U = scaling * (grad_W @ V_lora)
        grad_V = scaling * (grad_W.T @ U_lora)
        LoRA = scaling * (U_lora @ V_lora.transpose(-2, -1))
        W = R + LoRA
        grad_x = F.linear(grad_output, W.t())
        return grad_x, None, grad_U, grad_V, None, None


class GenoAttentionV4(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, rank=16, alpha=1.0, layer_id=0, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.rank = rank
        self.alpha = alpha
        D = hidden_dim
        d = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.R_q = SpectralR(D, D, 1000 + layer_id*6 + 0, d)
        self.R_k = SpectralR(D, D, 1000 + layer_id*6 + 1, d)
        self.R_v = SpectralR(D, D, 1000 + layer_id*6 + 2, d)
        self.R_o = SpectralR(D, D, 1000 + layer_id*6 + 3, d)

        self.U_Q = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_Q = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.U_K = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_K = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.U_V = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_V = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.U_O = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_O = nn.Parameter(torch.randn(D, rank) * 0.01)

    def forward(self, x):
        B, T, D = x.shape
        R_q = self.R_q(); R_k = self.R_k(); R_v = self.R_v(); R_o = self.R_o()

        Q = SpectralLinearFunction.apply(x, R_q, self.U_Q, self.V_Q, self.alpha, self.rank)
        K = SpectralLinearFunction.apply(x, R_k, self.U_K, self.V_K, self.alpha, self.rank)
        V = SpectralLinearFunction.apply(x, R_v, self.U_V, self.V_V, self.alpha, self.rank)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        out = SpectralLinearFunction.apply(out, R_o, self.U_O, self.V_O, self.alpha, self.rank)
        return out


class GenoFFNV4(nn.Module):
    def __init__(self, hidden_dim, ff_mult=4, rank=16, alpha=1.0, layer_id=0, device=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        D = hidden_dim; H = hidden_dim * ff_mult
        d = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.R_ff1 = SpectralR(H, D, 1000 + layer_id*6 + 4, d)
        self.R_ff2 = SpectralR(D, H, 1000 + layer_id*6 + 5, d)

        self.U_FF1 = nn.Parameter(torch.randn(H, rank) * 0.01)
        self.V_FF1 = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.U_FF2 = nn.Parameter(torch.randn(D, rank) * 0.01)
        self.V_FF2 = nn.Parameter(torch.randn(H, rank) * 0.01)

    def forward(self, x):
        R_ff1 = self.R_ff1(); R_ff2 = self.R_ff2()
        h = SpectralLinearFunction.apply(x, R_ff1, self.U_FF1, self.V_FF1, self.alpha, self.rank)
        h = F.relu(h)
        out = SpectralLinearFunction.apply(h, R_ff2, self.U_FF2, self.V_FF2, self.alpha, self.rank)
        return out


class GenoTransformerLayerV4(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, rank=16, alpha=1.0, layer_id=0, device=None):
        super().__init__()
        self.attn = GenoAttentionV4(hidden_dim, num_heads, rank, alpha, layer_id, device)
        self.ffn = GenoFFNV4(hidden_dim, 4, rank, alpha, layer_id, device)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GenoTransformerV4(nn.Module):
    def __init__(self, vocab_size=50000, hidden_dim=4096, num_layers=12,
                 num_heads=8, rank=16, alpha=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)

        self.layers = nn.ModuleList([
            GenoTransformerLayerV4(hidden_dim, num_heads, rank, alpha, i, d)
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
