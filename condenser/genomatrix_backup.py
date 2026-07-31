"""
GenoMatrix V3 FINAL: Full-rank ephemeral weights + LoRA modulation.

W_eff = R + U @ V^T
  R = random matrix from deterministic seed (full rank, ephemeral)
  U, V = learned low-rank factors from Genom (stored, trained)

For each Transformer-equivalent layer:
  W_Q = R_Q + U_Q @ V_Q^T
  W_K = R_K + U_K @ V_K^T
  W_V = R_V + U_V @ V_V^T
  W_O = R_O + U_O @ V_O^T
  W_FF1 = R_FF1 + U_FF1 @ V_FF1^T
  W_FF2 = R_FF2 + U_FF2 @ V_FF2^T

All R matrices are generated from seeds (not stored).
Only U, V factors are learned (stored, tiny).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EphemeralLinearFunction(torch.autograd.Function):
    """Linear layer with ephemeral weight: W = R + U@V^T.
    R generated from seed, never stored.
    U, V provided as parameters."""
    @staticmethod
    def forward(ctx, x, U, V, seed, in_features, out_features):
        ctx.seed = seed
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.save_for_backward(x, U, V)

        with torch.no_grad():
            gen = torch.Generator(device=x.device)
            gen.manual_seed(abs(seed) % (2**31 - 1))
            R = torch.randn(out_features, in_features, generator=gen,
                           device=x.device, dtype=x.dtype)
            R.div_(math.sqrt(in_features))

        # W_eff = R + U @ V^T
        LoRA = U @ V.transpose(-2, -1) if U.dim() > 1 else U.unsqueeze(-1) @ V.unsqueeze(-2)
        W = R + LoRA

        out = F.linear(x, W)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, U, V = ctx.saved_tensors
        seed = ctx.seed
        in_f = ctx.in_features
        out_f = ctx.out_features

        with torch.no_grad():
            gen = torch.Generator(device=grad_output.device)
            gen.manual_seed(abs(seed) % (2**31 - 1))
            R = torch.randn(out_f, in_f, generator=gen,
                           device=grad_output.device, dtype=grad_output.dtype)
            R.div_(math.sqrt(in_f))

        B = x.shape[0]
        x_flat = x.reshape(-1, in_f)
        go_flat = grad_output.reshape(-1, out_f)

        grad_W = go_flat.T @ x_flat
        grad_U = grad_W @ V
        grad_V = grad_W.T @ U

        W = R + U @ V.transpose(-2, -1)
        grad_x = F.linear(grad_output, W.t())

        return grad_x, grad_U, grad_V, None, None, None


class GenoAttention(nn.Module):
    """Multi-head attention with ephemeral weight generation."""
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

    def forward(self, x):
        B, T, D = x.shape

        Q = EphemeralLinearFunction.apply(
            x, self.U_Q, self.V_Q, self.seeds['q'], D, D)
        K = EphemeralLinearFunction.apply(
            x, self.U_K, self.V_K, self.seeds['k'], D, D)
        V = EphemeralLinearFunction.apply(
            x, self.U_V, self.V_V, self.seeds['v'], D, D)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        out = EphemeralLinearFunction.apply(
            out, self.U_O, self.V_O, self.seeds['o'], D, D)

        return out


class GenoFFN(nn.Module):
    """FFN with ephemeral weight generation."""
    def __init__(self, hidden_dim, ff_mult=4, rank=16, seed=0, layer_id=0):
        super().__init__()
        ff_dim = hidden_dim * ff_mult
        self.seed_ff1 = seed + layer_id * 6 + 4
        self.seed_ff2 = seed + layer_id * 6 + 5

        self.U_FF1 = nn.Parameter(torch.randn(ff_dim, rank) * 0.01)
        self.V_FF1 = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.U_FF2 = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.V_FF2 = nn.Parameter(torch.randn(ff_dim, rank) * 0.01)

    def forward(self, x):
        h = EphemeralLinearFunction.apply(
            x, self.U_FF1, self.V_FF1, self.seed_ff1,
            x.shape[-1], x.shape[-1] * 4)
        h = F.relu(h)
        out = EphemeralLinearFunction.apply(
            h, self.U_FF2, self.V_FF2, self.seed_ff2,
            h.shape[-1], x.shape[-1])
        return out


class GenoTransformerLayer(nn.Module):
    """Full transformer layer with all weights ephemeral."""
    def __init__(self, hidden_dim, num_heads=8, rank=16, seed=0, layer_id=0):
        super().__init__()
        self.attn = GenoAttention(hidden_dim, num_heads, rank, seed, layer_id)
        self.ffn = GenoFFN(hidden_dim, ff_mult=4, rank=rank, seed=seed, layer_id=layer_id)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GenoTransformer(nn.Module):
    """Full model: Transformer architecture with ephemeral weights."""
    def __init__(
        self,
        vocab_size=50000,
        hidden_dim=4096,
        num_layers=12,
        num_heads=8,
        rank=16,
    ):
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
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def count_trained(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def effective_params(self):
        d = self.hidden_dim
        per_layer = 12 * d * d  # Q,K,V,O,FF1,FF2 total dims
        return self.num_layers * per_layer

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
        eph = 6 * d * d * 2  # 6 weight matrices per layer (ephemeral peak)
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

    def memory_report(self):
        trained = self.count_trained()
        eff = self.effective_params()
        model_gb = trained * 2 / 1e9
        optim_gb = trained * 8 / 1e9
        eph_gb = 6 * self.hidden_dim * self.hidden_dim * 2 / 1e9
        return {
            'trained_m': trained / 1e6,
            'effective_b': eff / 1e9,
            'compression': eff / trained if trained else 0,
            'model_gb': model_gb,
            'optim_gb': optim_gb,
            'eph_gb': eph_gb,
            'total_gb': model_gb + optim_gb + eph_gb,
        }
