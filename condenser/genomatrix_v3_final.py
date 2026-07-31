"""
GenoMatrix V3 OPTIMIZED: R cached in forward, no regen in backward.
Layer-level R batching: all 6 R matrices generated in one CUDA call.
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


def generate_R_batch(seed, D, device, dtype):
    total = 12 * D * D
    gen = _get_gen(device)
    gen.manual_seed(abs(seed) % (2**31 - 1))
    all_R = torch.randn(total, generator=gen, device=device, dtype=dtype)
    scale = math.sqrt(D)
    off = 0
    s = D * D
    R_Q = all_R[off:off+s].view(D, D).div_(scale); off += s
    R_K = all_R[off:off+s].view(D, D).div_(scale); off += s
    R_V = all_R[off:off+s].view(D, D).div_(scale); off += s
    R_O = all_R[off:off+s].view(D, D).div_(scale); off += s
    s = 4 * D * D
    R_FF1 = all_R[off:off+s].view(4*D, D).div_(scale); off += s
    R_FF2 = all_R[off:off+s].view(D, 4*D).div_(scale)
    return {'q': R_Q, 'k': R_K, 'v': R_V, 'o': R_O, 'ff1': R_FF1, 'ff2': R_FF2}


class EphemeralLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, R, U, V):
        U = U.to(x.dtype)
        V = V.to(x.dtype)
        LoRA = U @ V.transpose(-2, -1)
        W = R + LoRA
        out = F.linear(x, W)
        ctx.save_for_backward(x, R, U, V)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, R, U, V = ctx.saved_tensors
        dt = grad_output.dtype
        R = R.to(dt)
        U = U.to(dt)
        V = V.to(dt)
        x = x.to(dt)
        in_f = x.shape[-1]
        out_f = grad_output.shape[-1]

        x_flat = x.reshape(-1, in_f)
        go_flat = grad_output.reshape(-1, out_f)

        grad_W = go_flat.T @ x_flat
        grad_U = grad_W @ V
        grad_V = grad_W.T @ U

        LoRA = U @ V.transpose(-2, -1)
        W = R + LoRA
        grad_x = F.linear(grad_output, W.t())

        return grad_x, None, grad_U, grad_V


class GenoAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, rank=16, seed=0, layer_id=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.rank = rank

        seeds = {
            'q': seed + layer_id * 6,
            'k': seed + layer_id * 6 + 1,
            'v': seed + layer_id * 6 + 2,
            'o': seed + layer_id * 6 + 3,
        }
        self.seeds = seeds

        self.U_Q = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.V_Q = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.U_K = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.V_K = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.U_V = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.V_V = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.U_O = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.V_O = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)

    def forward(self, x, Rs):
        B, T, D = x.shape

        Q = EphemeralLinearFunction.apply(x, Rs['q'], self.U_Q, self.V_Q)
        K = EphemeralLinearFunction.apply(x, Rs['k'], self.U_K, self.V_K)
        V = EphemeralLinearFunction.apply(x, Rs['v'], self.U_V, self.V_V)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        out = EphemeralLinearFunction.apply(out, Rs['o'], self.U_O, self.V_O)

        return out


class GenoFFN(nn.Module):
    def __init__(self, hidden_dim, ff_mult=4, rank=16, seed=0, layer_id=0):
        super().__init__()
        self.seed_ff1 = seed + layer_id * 6 + 4
        self.seed_ff2 = seed + layer_id * 6 + 5
        hidden_dim_ = hidden_dim
        self.U_FF1 = nn.Parameter(torch.randn(hidden_dim * ff_mult, rank) * 0.01)
        self.V_FF1 = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.U_FF2 = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.V_FF2 = nn.Parameter(torch.randn(hidden_dim * ff_mult, rank) * 0.01)

    def forward(self, x, Rs):
        h = EphemeralLinearFunction.apply(x, Rs['ff1'], self.U_FF1, self.V_FF1)
        h = F.relu(h)
        out = EphemeralLinearFunction.apply(h, Rs['ff2'], self.U_FF2, self.V_FF2)
        return out


class GenoTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, rank=16, seed=0, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.seed = seed
        self.attn = GenoAttention(hidden_dim, num_heads, rank, seed, layer_id)
        self.ffn = GenoFFN(hidden_dim, 4, rank, seed, layer_id)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        D = x.shape[-1]
        Rs = generate_R_batch(self.seed + self.layer_id * 6, D, x.device, x.dtype)
        x = x + self.attn(self.norm1(x), Rs)
        x = x + self.ffn(self.norm2(x), Rs)
        return x


class GenoTransformer(nn.Module):
    def __init__(self, vocab_size=50000, hidden_dim=4096, num_layers=12,
                 num_heads=8, rank=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)

        self.layers = nn.ModuleList([
            GenoTransformerLayer(hidden_dim, num_heads, rank, seed=42, layer_id=i)
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

    @staticmethod
    def scaling_analysis(vocab_size=50000, hidden_dim=4096, num_layers=2480, rank=1):
        d = hidden_dim; L = num_layers; r = rank
        stored_per_layer = 0
        for out_d, in_d in [(d,d),(d,d),(d,d),(d,d),(4*d,d),(d,4*d)]:
            stored_per_layer += r * (out_d + in_d)
        stored = L * stored_per_layer
        embedding = vocab_size * d
        total_stored = stored + embedding
        effective = L * 12 * d * d
        eph = 6 * d * d * 2
        model_mb = total_stored * 2 / 1e6
        optim_mb = total_stored * 2 / 1e6
        eph_mb = eph / 1e6
        total_mb = model_mb + optim_mb + eph_mb
        return {
            'total_stored': total_stored,
            'effective': effective,
            'compression': effective / max(total_stored, 1),
            'model_mb': model_mb,
            'optim_mb': optim_mb,
            'eph_mb': eph_mb,
            'total_mb': total_mb,
            'fits_6gb': total_mb < 6000,
        }
