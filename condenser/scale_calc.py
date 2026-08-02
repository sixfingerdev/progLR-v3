"""
6GB VRAM scale calculator, calibrated to measured points (RTX 3060, fp32, WT103):
  ours      (V3-Embed + analytic head + ckpt, B=16): peak = 16B*P + 4B*V*D + 1.45GB
             calib: ProgLR D=1024 28.8M -> 1.99GB (OH 1.32), ProgLR D=1536 61.7M -> 2.79 (OH 1.50),
                    dense D=1024 102.6M -> 3.33GB (OH 1.48)
  vanilla   (nn.Embedding + dense nn.Linear head, autograd, no ckpt, B=8):
             peak = 16B*P + 8B*B*SEQ*V (logits + saved softmax) + 2.5GB
             calib: D=512 L=8 V=50K B=8 76.9M -> 5.04GB
Prints max D and max trainable params that fit 6GB per architecture and vocab.
"""
import os

GB = 1e9
SEQ = 512

def p_ours(V, D, L, n_std=2, r=16):
    n_lr = L - n_std
    pos = 1024 * D
    norm = 2 * L * D + D
    return n_std * 12 * D * D + n_lr * 18 * D * r + 2 * r * (V + D) + pos + norm

def p_dense_v3(V, D, L, r=16):
    return L * 12 * D * D + 2 * r * (V + D) + 1024 * D + 2 * L * D + D

def p_vanilla(V, D, L):
    return L * 12 * D * D + 2 * V * D + 1024 * D + 2 * L * D + D

def mem_ours(P, V, D):
    return 16 * P + 4 * V * D + 1.45 * GB

def mem_dense_v3(P, V, D):
    return 16 * P + 4 * V * D + 1.45 * GB

def mem_vanilla(P, V, D, B=8):
    return 16 * P + 8 * B * SEQ * V + 2.5 * GB

D_GRID = [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096, 4608, 5120]
BUDGET = 6 * GB

def max_fit(p_fn, mem_fn, V, L, **kw):
    best = None
    for D in D_GRID:
        P = p_fn(V, D, L, **kw)
        m = mem_fn(P, V, D)
        if m <= BUDGET:
            best = (D, P, m)
    return best

print(f"{'vocab':>8} {'L':>3} | {'vanilla dense head':>24} | {'dense + our head':>22} | {'OUR ProgLR stack':>24}")
print('-' * 95)
for V, L in [(50000, 8), (50000, 24), (250000, 8), (250000, 24), (500000, 8), (500000, 24)]:
    row = []
    for kind, p_fn, m_fn, kw in [('vanilla', p_vanilla, mem_vanilla, {}),
                                 ('dense_v3', p_dense_v3, mem_dense_v3, {}),
                                 ('ours', p_ours, mem_ours, {})]:
        r = max_fit(p_fn, m_fn, V, L, **kw)
        if r is None:
            row.append('NONE fits')
        else:
            D, P, m = r
            row.append(f'D={D} {P/1e6:.0f}M ({m/GB:.1f}GB)')
    print(f'{V/1000:>7.0f}K {L:>3} | {row[0]:>24} | {row[1]:>22} | {row[2]:>24}')

print()
print('Calibration sanity: ours D=1024 L=8 V=50K ->', f"{mem_ours(p_ours(50000,1024,8),50000,1024)/GB:.2f}GB (measured 1.99)")
print('                      ours D=1536 L=8 V=50K ->', f"{mem_ours(p_ours(50000,1536,8),50000,1536)/GB:.2f}GB (measured 2.79)")
print('                      dense_v3 D=1024 L=8    ->', f"{mem_dense_v3(p_dense_v3(50000,1024,8),50000,1024)/GB:.2f}GB (measured 3.33)")
print('                      vanilla D=512 L=8 B=8  ->', f"{mem_vanilla(p_vanilla(50000,512,8),50000,512)/GB:.2f}GB (measured 5.04)")
