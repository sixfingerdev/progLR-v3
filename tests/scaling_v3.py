"""Scaling analysis: V3 GenoTransformer 500B in 6GB."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from condenser.genomatrix import GenoTransformer

print("=" * 65)
print("V3 GenoTransformer: Scaling Analysis")
print("=" * 65)

configs = [
    ("Small  (proof)",    512,   24,   8,  50000),
    ("Medium (quality)",  1024,  48,   4,  50000),
    ("Large  (scaling)",  2048,  96,   2,  50000),
    ("XLarge (target)",   4096,  2480, 1,  50000),
]

for label, D, L, rank, V in configs:
    if D * D * L > 1e11:
        print(f"\n{label}: D={D}, L={L}, rank={rank}")
        print("  [estimating, may be slow for large numbers]")
    s = GenoTransformer.scaling_analysis(V, D, L, rank)
    eff_b = s['effective'] / 1e9
    fit = "[OK]" if s['fits_6gb'] else "[FAIL]"

    print(f"\n{label}: D={D}, L={L}, rank={rank}  {fit}")
    print(f"  Trained:   {s['total_stored']/1e6:>9.1f}M")
    print(f"  Effective: {eff_b:>9.2f}B  ({s['compression']:.0f}x)")
    print(f"  Model:     {s['model_mb']:>9.0f}MB  (fp16)")
    print(f"  Optimizer: {s['optim_mb']:>9.0f}MB  (8-bit Adam)")
    print(f"  Ephemeral: {s['eph_mb']:>9.0f}MB  (peak 1 layer)")
    print(f"  Total:     {s['total_mb']:>9.0f}MB  {'< 6GB = WIN!' if s['fits_6gb'] else 'EXCEEDS 6GB'}")
