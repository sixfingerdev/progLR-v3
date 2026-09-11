import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import time
import math

from condenser.layer import CondenserModel
from condenser.baseline_transformer import BaselineTransformer

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_vram_usage(device):
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / (1024**2)
    return 0.0

def make_batch(vocab_size, batch_size, seq_len, device):
    return torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

def measure_vram(model, batch_size, seq_len, vocab_size, device, num_steps=5):
    model = model.to(device)
    model.train()

    batch = make_batch(vocab_size, batch_size, seq_len, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    start_mem = torch.cuda.memory_allocated(device)

    for step in range(num_steps):
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
        loss.backward()
        if step == 0:
            peak_mem = torch.cuda.max_memory_allocated(device)
        batch = make_batch(vocab_size, batch_size, seq_len, device)

    end_mem = torch.cuda.memory_allocated(device)
    peak_mib = peak_mem / (1024**2)

    return {
        'start_mib': start_mem / (1024**2),
        'peak_mib': peak_mem / (1024**2),
        'end_mib': end_mem / (1024**2),
    }

def train_step(model, batch, optimizer, vocab_size):
    logits = model(batch)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def benchmark_models(device='cuda', hidden_dim=128, num_layers=4, seq_len=64,
                     n_atoms_in_pool=10000, atom_dim=32, n_selected=16):
    vocab_size = 500

    condenser = CondenserModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        atom_dim=atom_dim,
        n_atoms_in_pool=n_atoms_in_pool,
        n_selected=n_selected,
        genom_hidden=hidden_dim,
    )

    transformer = BaselineTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=4,
    )

    c_params = count_trainable_params(condenser)
    t_params = count_trainable_params(transformer)

    print("=" * 60)
    print(f"CONFIG: hidden_dim={hidden_dim}, layers={num_layers}, seq_len={seq_len}")
    print(f"        atom_dim={atom_dim}, pool={n_atoms_in_pool}, k={n_selected}")
    print("-" * 60)

    print(f"\n{'Model':<20} {'Params':<15} {'Ratio':<15}")
    print("-" * 50)
    t_weight_params = num_layers * (hidden_dim * hidden_dim * 2)
    c_overhead = c_params - num_layers * atom_dim * 2
    print(f"{'Condenser':<20} {c_params:<15,} {'':<15}")
    print(f"{'Transformer':<20} {t_params:<15,} {'':<15}")
    eff_w = num_layers * hidden_dim * hidden_dim
    print(f"{'Effective W only':<20} {eff_w:<15,} {'':<15}")
    print(f"{'Compression':<20} {'':<15} {t_params/c_params:.2f}x smaller")

    print("\n--- VRAM Benchmark ---")
    for bs in [1, 2, 4]:
        results = {}
        for name, model in [('Condenser', condenser), ('Transformer', transformer)]:
            try:
                m = measure_vram(model, bs, seq_len, vocab_size, device, num_steps=3)
                results[name] = m
            except RuntimeError as e:
                results[name] = {'peak_mib': float('inf'), 'error': str(e)}

        line = f"Batch {bs:>2}:  "
        for name in ['Condenser', 'Transformer']:
            r = results[name]
            if r.get('peak_mib', 0) == float('inf'):
                line += f"{name}: OOM  "
            else:
                line += f"{name}: {r['peak_mib']:.0f}MB peak  "
        print(line)

        if bs == 1 and device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n--- Training Speed ---")
    optim_c = torch.optim.AdamW(condenser.parameters(), lr=3e-4)
    optim_t = torch.optim.AdamW(transformer.parameters(), lr=3e-4)

    for bs in [4]:
        batch = make_batch(vocab_size, bs, seq_len, device)
        for name, model, optim in [('Condenser', condenser, optim_c), ('Transformer', transformer, optim_t)]:
            model = model.to(device)
            optim.zero_grad()

            start = time.time()
            losses = []
            for _ in range(10):
                l = train_step(model, batch, optim, vocab_size)
                losses.append(l)
                batch = make_batch(vocab_size, bs, seq_len, device)

            elapsed = time.time() - start
            avg_loss = sum(losses) / len(losses)
            print(f"  {name:<15} {elapsed/10*1000:>6.1f}ms/step   loss={avg_loss:.4f}")

    return {'condenser_params': c_params, 'transformer_params': t_params}


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    benchmark_models(
        device=device,
        hidden_dim=128,
        num_layers=4,
        seq_len=64,
        n_atoms_in_pool=10000,
        atom_dim=32,
        n_selected=16,
    )

    print("\n" + "=" * 60)
    print("Scaling: larger model (hidden_dim=256, 6 layers)")
    print("=" * 60)
    benchmark_models(
        device=device,
        hidden_dim=256,
        num_layers=6,
        seq_len=128,
        n_atoms_in_pool=50000,
        atom_dim=64,
        n_selected=32,
    )
