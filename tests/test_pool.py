import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from condenser.pool import AtomPool

def test_atom_pool_creation():
    pool = AtomPool(n_atoms=1000, atom_dim=64, seed=42)
    assert pool.pool.shape == (1000, 64)
    assert not torch.isnan(pool.pool).any()
    norms = pool.pool.norm(dim=1)
    assert (norms - 1.0).abs().max() < 1e-5
    print(f"[PASS] Pool: {pool.pool.shape}, unit-norm atoms")

def test_query():
    pool = AtomPool(n_atoms=1000, atom_dim=64, seed=42)
    code = torch.randn(4, 64)
    scores = pool.query(code)
    assert scores.shape == (4, 1000)
    assert not torch.isnan(scores).any()
    print(f"[PASS] Query: {scores.shape}")

def test_soft_sample():
    pool = AtomPool(n_atoms=100, atom_dim=16, seed=42)
    code = torch.randn(2, 16)
    scores = pool.query(code)
    weights = pool.sample(scores, k=10, tau=0.5, hard=False)
    assert weights.shape == (2, 100)
    assert (weights.sum(dim=-1) - 1.0).abs().max() < 1e-5
    print(f"[PASS] Soft sample: {weights.shape}, sums ~1.0")

def test_hard_sample():
    pool = AtomPool(n_atoms=100, atom_dim=16, seed=42)
    code = torch.randn(2, 16)
    scores = pool.query(code)
    weights = pool.sample(scores, k=10, tau=0.5, hard=True)
    assert weights.shape == (2, 100)
    active_per_row = (weights > 0).sum(dim=-1)
    assert (active_per_row > 0).all()
    print(f"[PASS] Hard sample: ~{active_per_row.float().mean().item():.1f} active atoms/row (hard Gumbel-Softmax)")

def test_sample_differentiable():
    pool = AtomPool(n_atoms=50, atom_dim=8, seed=42)
    code = torch.randn(2, 8, requires_grad=True)
    scores = pool.query(code)
    weights = pool.sample(scores, k=5, tau=0.5, hard=True)
    loss = weights.sum()
    loss.backward()
    assert code.grad is not None
    assert not torch.isnan(code.grad).any()
    print(f"[PASS] Gradients flow: grad shape {code.grad.shape}")

def test_synthesize_weight():
    pool = AtomPool(n_atoms=100, atom_dim=16, seed=42)

    def expand_fn(atoms, out_dim, in_dim):
        base = torch.randn(atoms.shape[-1], out_dim, in_dim, device=atoms.device)
        return (atoms.unsqueeze(-1).unsqueeze(-1) * base.unsqueeze(0)).sum(dim=1)

    code = torch.randn(3, 16)
    scores = pool.query(code)
    weights = pool.sample(scores, k=16, tau=0.5, hard=False)
    W = pool.synthesize_weight(weights, expand_fn, 32, 64)
    assert W.shape == (3, 32, 64)
    print(f"[PASS] Weight synthesis: {W.shape}")

def test_full_differentiable_pipeline():
    pool = AtomPool(n_atoms=100, atom_dim=16, seed=42)

    def expand_fn(atoms, out_dim, in_dim):
        base = torch.randn(atoms.shape[-1], out_dim, in_dim, device=atoms.device)
        return (atoms.unsqueeze(-1).unsqueeze(-1) * base.unsqueeze(0)).sum(dim=1)

    code = torch.randn(2, 16, requires_grad=True)
    scores = pool.query(code)
    weights = pool.sample(scores, k=10, tau=0.5, hard=True)
    W = pool.synthesize_weight(weights, expand_fn, 16, 16)
    out = (W @ torch.randn(2, 16, 1)).sum()
    out.backward()
    assert code.grad is not None
    assert not torch.isnan(code.grad).any()
    print(f"[PASS] End-to-end differentiable: grad norm {code.grad.norm():.4f}")

if __name__ == "__main__":
    print("=== AtomPool Tests ===\n")
    test_atom_pool_creation()
    test_query()
    test_soft_sample()
    test_hard_sample()
    test_sample_differentiable()
    test_synthesize_weight()
    test_full_differentiable_pipeline()
    print("\n All AtomPool tests passed!")
