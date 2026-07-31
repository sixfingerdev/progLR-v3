"""
V3-All VRAM Scaling Projection for 6GB GPU.
Finds max configuration that fits in 6GB VRAM.
"""
import math

DIMS = [2048, 3072, 4096, 5120, 6144, 8192]
LAYERS = [8, 12, 16, 24]
VOCAB = 500_000
RANK = 8
STD_LAYERS = 2

def embed_params(D):
    return VOCAB * RANK + D * RANK

def std_layer_params(D):
    return 12 * D * D

def lr_layer_params(D):
    return 18 * D * RANK

for D in DIMS:
    for L in LAYERS:
        n_lr = L - STD_LAYERS
        if n_lr < 0:
            continue
        total = embed_params(D) + 512 * D + STD_LAYERS * std_layer_params(D) + n_lr * lr_layer_params(D) + 2 * D
        model_mb = total * 2 / 1e6
        optim_mb = total * 8 / 1e6
        act_mb = D * 512 * 4 * 2 / 1e6
        total_gb = (model_mb + optim_mb + act_mb) / 1000
        eff = STD_LAYERS + n_lr * 0.8
        fits = "YES" if total_gb < 6.0 else "NO "

        print(f'D={D:<5} L={L:<2} ({STD_LAYERS}Std+{n_lr}LR) | '
              f'params={total/1e6:<6.1f}M | '
              f'model={model_mb:<5.0f}MB optim={optim_mb:<5.0f}MB act={act_mb:<5.0f}MB | '
              f'total={total_gb:<4.1f}GB | fits={fits} | eff~{eff:.0f}L')
