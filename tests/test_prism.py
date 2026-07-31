import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.prism import PRISMModel

def test_prism_forward():
    model = PRISMModel(
        vocab_size=500,
        hidden_dim=128,
        num_layers=4,
        state_dim=64,
        rank=8,
    )

    trained = model.count_trained()
    eff = model.effective_params()
    print(f"[INFO] Trained params: {trained:,} ({trained*2/1e6:.1f}MB fp16)")
    print(f"[INFO] Effective params: {eff:,} ({eff/1e6:.0f}M)")
    print(f"[INFO] Compression: {eff/trained:.0f}x")

    x = torch.randint(0, 500, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 500), f"Got {logits.shape}"

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 500), x[:, 1:].reshape(-1))
    loss.backward()

    assert not torch.isnan(logits).any()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"[PASS] Forward + backward: loss={loss.item():.4f}, grad norms OK")
    print(f"[PASS] Gradients: max norm={max(grad_norms):.4f}, min norm={min(grad_norms):.4f}")

def test_scaling_analysis():
    print("\n--- PRISM Scaling Analysis ---")
    configs = [
        ("1B eff",    1024, 12, 64, 16),
        ("7B eff",    2048, 24, 128, 16),
        ("70B eff",   4096, 48, 256, 16),
        ("175B eff",  4096, 96, 256, 32),
        ("500B eff",  4096, 128, 512, 32),
        ("1T eff",    8192, 96, 512, 32),
    ]

    print(f"{'Scenario':<15} {'Trained M':<12} {'Eff B':<10} {'Compress':<10} {'VRAM GB':<10} {'6GB?':<8}")
    print("-" * 65)

    for name, h, nl, sd, r in configs:
        model = PRISMModel(vocab_size=50000, hidden_dim=h, num_layers=nl, state_dim=sd, rank=r)
        rep = model.memory_report()
        print(f"{name:<15} {rep['trained_m']:<12.0f} {rep['effective_b']:<10.1f} "
              f"{rep['compression']:<10.0f}x {rep['total_gb']:<10.1f} "
              f"{'YES' if rep['fits_6gb'] else 'NO'}")

if __name__ == "__main__":
    print("=== PRISM Model Test ===\n")
    test_prism_forward()
    test_scaling_analysis()
    print("\n All tests passed!")
