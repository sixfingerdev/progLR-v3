"""Shared RoPE helper: single source for condenser + proglr (no duplication)."""
import math
import torch


def build_rope_cache(max_pos, head_dim, base=10000.0, device=None, dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(device=device, dtype=dtype), torch.sin(freqs).to(device=device, dtype=dtype)


def apply_rope(q, k, cos, sin):
    """q,k: (B,H,T,dh). cos/sin: (T, dh/2). Rotary on last dim pairs."""
    T = q.shape[2]
    c, s = cos[:T].to(q.device, q.dtype), sin[:T].to(q.device, q.dtype)
    # (1,1,T,dh/2)
    c = c.unsqueeze(0).unsqueeze(0)
    s = s.unsqueeze(0).unsqueeze(0)
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    qr = torch.stack([-q2, q1], dim=-1).flatten(-2)
    kr = torch.stack([-k2, k1], dim=-1).flatten(-2)
    cc = torch.stack([c, c], dim=-1).flatten(-2)
    ss = torch.stack([s, s], dim=-1).flatten(-2)
    return q * cc + qr * ss, k * cc + kr * ss
