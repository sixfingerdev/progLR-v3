"""ProgLR model: V3-Embed + stacked Std/LR transformer layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

try:
    from condenser.v3_embed import V3WeightEmbedding
except ImportError:  # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from condenser.v3_embed import V3WeightEmbedding


class LRLayer(nn.Module):
    """Low-rank attention + FFN (factorized W = (a/r) U V^T, no full matrix materialized)."""

    def __init__(self, dim, heads, r=16, a=64):
        super().__init__()
        self.h, self.dh, self.r, self.a = heads, dim // heads, r, a
        for nm, od, id_ in [('q', dim, dim), ('k', dim, dim), ('v', dim, dim),
                            ('o', dim, dim), ('ff1', 4 * dim, dim), ('ff2', dim, 4 * dim)]:
            setattr(self, f'U_{nm}', nn.Parameter(torch.randn(od, r) * 0.02))
            setattr(self, f'V_{nm}', nn.Parameter(torch.randn(id_, r) * 0.02))
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def _l(self, x, nm):
        s = self.a / self.r
        return s * ((x @ getattr(self, f'V_{nm}')) @ getattr(self, f'U_{nm}').t())

    def forward(self, x):
        B, T, D = x.shape; h = self.dh; n = self.n1(x)
        Q = self._l(n, 'q').view(B, T, self.h, h).transpose(1, 2)
        K = self._l(n, 'k').view(B, T, self.h, h).transpose(1, 2)
        V_ = self._l(n, 'v').view(B, T, self.h, h).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V_, is_causal=True)
        x = x + self._l(a.transpose(1, 2).reshape(B, T, D), 'o')
        n2 = self.n2(x)
        return x + self._l(F.relu(self._l(n2, 'ff1')), 'ff2')


class StdLayer(nn.Module):
    """Standard dense transformer layer."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.dh = dim // heads
        self.W_Q = nn.Linear(dim, dim, bias=False); self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False); self.W_O = nn.Linear(dim, dim, bias=False)
        self.W1 = nn.Linear(dim, dim * 4, bias=False); self.W2 = nn.Linear(dim * 4, dim, bias=False)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape; h = self.h; dh = self.dh; n = self.n1(x)
        Q = self.W_Q(n).view(B, T, h, dh).transpose(1, 2)
        K = self.W_K(n).view(B, T, h, dh).transpose(1, 2)
        V = self.W_V(n).view(B, T, h, dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        x = x + self.W_O(a.transpose(1, 2).reshape(B, T, D))
        n = self.n2(x)
        return x + self.W2(F.relu(self.W1(n)))


class ProgLR(nn.Module):
    """V3-Embed + n_std standard layers + n_lr low-rank layers + final LayerNorm.

    Head is analytic: U/V/R are exposed as buffers/parameters so gradients are computed
    outside the module (see proglr.head.head_loss_fn).
    """

    def __init__(self, vs, D, H, n_std, n_lr, r=16, a=64, max_pos=1024, checkpoint_all=False):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1, max_pos, D) * 0.01)
        self.checkpoint_all = checkpoint_all
        self.layers = nn.ModuleList()
        for _ in range(n_std):
            self.layers.append(StdLayer(D, H))
        for _ in range(n_lr):
            self.layers.append(LRLayer(D, H, r, a))
        self.norm = nn.LayerNorm(D)

    def get_hidden(self, x):
        B, T = x.shape
        h = self.emb.forward_embed(x) + self.pos[:, :T]
        for l in self.layers:
            if self.checkpoint_all:
                h = ckpt.checkpoint(l, h, use_reentrant=False)
            else:
                h = l(h)
        return self.norm(h)
