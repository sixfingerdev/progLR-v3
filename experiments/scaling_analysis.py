import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

def condenser_memory(hidden_dim, num_layers, atom_dim, n_atoms, genom_hidden):
    vocab_size = 50000
    embed_params = vocab_size * hidden_dim
    genom_params = (hidden_dim + 1) * genom_hidden + genom_hidden * atom_dim
    norm_params = hidden_dim * 2 * num_layers

    total_trainable = embed_params + genom_params * num_layers + norm_params

    buffer_size = (hidden_dim * atom_dim + atom_dim * hidden_dim + hidden_dim) * num_layers
    pool_size = n_atoms * atom_dim

    trainable_mb = total_trainable * 2 / (1024**2)
    buffer_mb = (buffer_size + pool_size) * 2 / (1024**2)

    effective_params = num_layers * (hidden_dim * hidden_dim * 2)

    return {
        'trainable_params': total_trainable,
        'trainable_mb': trainable_mb,
        'buffer_mb': buffer_mb,
        'total_mb': trainable_mb + buffer_mb,
        'effective_params': effective_params,
    }


def transformer_memory(hidden_dim, num_layers, num_heads=8, vocab_size=50000):
    embed_params = vocab_size * hidden_dim
    layer_params = (3 * hidden_dim * hidden_dim) + (hidden_dim * hidden_dim) + \
                   (hidden_dim * 4 * hidden_dim) + (4 * hidden_dim * hidden_dim) + \
                   (2 * hidden_dim * 2)
    total = embed_params + layer_params * num_layers + hidden_dim * 2
    return {'params': total, 'memory_mb': total * 2 / (1024**2)}


scenarios = [
    ('Tiny (7M eff)',      256,   6,  64,   50000,   256),
    ('Small (100M eff)',   1024,  12, 128,  100000,  1024),
    ('Medium (1B eff)',    2048,  24, 256,  500000,  2048),
    ('Large (7B eff)',     4096,  32, 512,  1000000, 4096),
    ('XL (70B eff)',       8192,  48, 1024, 10000000, 8192),
    ('XXL (500B eff)',     16384, 64, 2048, 50000000, 16384),
    ('ULTRA (1T eff)',     24576, 80, 3072, 100000000, 24576),
]

print("=" * 150)
print(f"{'Scenario':<30} {'Eff W Params':<15} {'C Trainable':<15} {'C Total Mem':<15} {'T Params':<15} {'T Memory':<15} {'C/T Ratio':<12} {'Mem Factor':<12} {'6GB OK?'}")
print("=" * 150)

for name, h, nl, ad, na, gh in scenarios:
    c = condenser_memory(h, nl, ad, na, gh)
    t = transformer_memory(h, nl)

    ratio = t['params'] / c['trainable_params']
    mem_factor = t['memory_mb'] / c['total_mb']
    fits_in_6gb = c['total_mb'] < 6000

    print(f"{name:<30} {c['effective_params']/1e6:<15.0f}M "
          f"{c['trainable_params']/1e6:<15.2f}M "
          f"{c['total_mb']:<15.1f} "
          f"{t['params']/1e6:<15.0f}M "
          f"{t['memory_mb']:<15.0f} "
          f"{ratio:<12.1f}x "
          f"{mem_factor:<12.1f}x "
          f"{'YES' if fits_in_6gb else 'NO'}")

    if c['total_mb'] > 6000 and c['trainable_mb'] < 6000:
        print(f"  -> Buffer too large ({c['buffer_mb']:.0f}MB), but trainable is only {c['trainable_mb']:.0f}MB")
        print(f"  -> Solution: stream atoms from disk instead of storing in VRAM")

print()
print("=" * 150)
print(f"{'Key insight':<30} {'Buffer memory dominates at scale, but is FIXED (not trainable)'}")
print(f"{'':<30} {'Trainable params are 10-100x smaller than effective weights'}")
print(f"{'':<30} {'Buffers can be offloaded to CPU/disk with memory-mapped I/O'}")
print("=" * 150)
