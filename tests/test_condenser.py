import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from condenser.layer import CondenserModel, CondenserLayer

def test_forward_pass():
    model = CondenserModel(
        vocab_size=500,
        hidden_dim=64,
        num_layers=3,
        atom_dim=16,
        n_atoms_in_pool=1000,
        n_selected=8,
        genom_hidden=64,
    )
    x = torch.randint(0, 500, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 500), f"Expected (2,32,500), got {logits.shape}"
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 500), x[:, 1:].reshape(-1))
    loss.backward()
    assert not torch.isnan(logits).any()
    print(f"[PASS] Forward + backward: loss={loss.item():.4f}")
    print(f"[INFO] Trainable params: {model.count_trainable_params():,}")
    print(f"[INFO] Buffers: {model.count_buffers():,}")
    mem = model.memory_usage_mb()
    print(f"[INFO] Trainable memory: {mem['trainable_params_mb']:.2f} MB")
    print(f"[INFO] Buffer memory: {mem['buffers_mb']:.2f} MB")

    n_layers = model.num_layers
    h = model.hidden_dim
    effective_weights = n_layers * h * h
    ratio = effective_weights / model.count_trainable_params()
    print(f"[INFO] Live params: {model.count_trainable_params():,} (stored in VRAM)")
    print(f"[INFO] Buffer atoms: {model.count_buffers():,} (fixed)")
    print(f"[RESULT] Effective W / stored params ratio: {ratio:.1f}x")

import torch.nn.functional as F

if __name__ == "__main__":
    print("=== Condenser Model Tests ===\n")
    test_forward_pass()
    print("\nAll tests passed!")
