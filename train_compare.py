import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import time
import math

from condenser.layer import CondenserModel
from condenser.baseline_transformer import BaselineTransformer


def generate_text_data(num_tokens=50000, vocab_size=500, seed=42):
    torch.manual_seed(seed)
    data = torch.randint(1, vocab_size, (num_tokens,))

    patterns = [
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 20, 30],
    ]
    for _ in range(200):
        pat = patterns[_ % len(patterns)]
        pos = torch.randint(0, num_tokens - len(pat), (1,)).item()
        data[pos:pos+len(pat)] = torch.tensor(pat)

    return data


def get_batches(data, batch_size, seq_len):
    n = len(data)
    num_steps = n // (batch_size * seq_len)
    data = data[:num_steps * batch_size * seq_len]
    data = data.view(batch_size, -1)
    for i in range(0, data.size(1) - seq_len, seq_len // 2):
        x = data[:, i:i+seq_len]
        yield x


def train_model(model, data, vocab_size, device, num_steps=500,
                lr=3e-4, batch_size=4, seq_len=64, model_name="Model"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    ppls = []
    vram_usage = []
    step_times = []
    step = 0

    while step < num_steps:
        for batch in get_batches(data, batch_size, seq_len):
            if step >= num_steps:
                break

            batch = batch.to(device)
            start = time.time()

            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            elapsed = time.time() - start

            losses.append(loss.item())
            ppl = math.exp(loss.item())
            ppls.append(ppl)
            step_times.append(elapsed)

            if device.type == 'cuda':
                vram_usage.append(torch.cuda.memory_allocated(device) / (1024**2))

            step += 1

            if step % 100 == 0:
                avg_loss = sum(losses[-50:]) / min(50, len(losses))
                avg_ppl = math.exp(avg_loss)
                avg_vram = sum(vram_usage[-50:]) / min(50, len(vram_usage)) if vram_usage else 0
                print(f"  {model_name} step {step:>4}: loss={avg_loss:.4f} ppl={avg_ppl:.2f} "
                      f"vram={avg_vram:.0f}MB speed={elapsed*1000:.0f}ms/step")

    return {
        'losses': losses,
        'ppls': ppls,
        'vram': vram_usage,
        'times': step_times,
        'final_loss': losses[-1],
        'best_ppl': min(ppls) if ppls else float('inf'),
        'avg_vram': sum(vram_usage[-100:]) / min(100, len(vram_usage)) if vram_usage else 0,
        'avg_step_ms': sum(step_times[-100:]) / min(100, len(step_times)) * 1000,
    }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    VOCAB_SIZE = 500
    HIDDEN_DIM = 256
    NUM_LAYERS = 6
    SEQ_LEN = 128
    BATCH_SIZE = 4
    NUM_STEPS = 1000
    ATOM_DIM = 64
    N_ATOMS = 50000
    N_SELECTED = 32

    print(f"\nConfig: vocab={VOCAB_SIZE}, hidden={HIDDEN_DIM}, layers={NUM_LAYERS}")
    print(f"        seq={SEQ_LEN}, batch={BATCH_SIZE}, steps={NUM_STEPS}")
    print(f"        atoms={N_ATOMS}, atom_dim={ATOM_DIM}, k={N_SELECTED}\n")

    data = generate_text_data(num_tokens=100000, vocab_size=VOCAB_SIZE)

    condenser = CondenserModel(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        atom_dim=ATOM_DIM,
        n_atoms_in_pool=N_ATOMS,
        n_selected=N_SELECTED,
        genom_hidden=HIDDEN_DIM,
    )

    transformer = BaselineTransformer(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=8,
    )

    c_params = sum(p.numel() for p in condenser.parameters() if p.requires_grad)
    t_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    eff_w = NUM_LAYERS * HIDDEN_DIM * HIDDEN_DIM

    print(f"{'Model':<20} {'Params':<12} {'VRAM (peak)':<12}")
    print("-" * 44)
    print(f"{'Condenser':<20} {c_params:<12,} {condenser.memory_usage_mb()['trainable_params_mb']+condenser.memory_usage_mb()['buffers_mb']:<10.1f} MB")
    print(f"{'Transformer':<20} {t_params:<12,} {'N/A':<12}")
    print(f"{'Effective W':<20} {eff_w:<12,}")
    print(f"{'Compression':<20} {t_params/c_params:.2f}x smaller")
    print()

    print("=" * 60)
    print("TRAINING CONDENSER")
    print("=" * 60)
    torch.cuda.reset_peak_memory_stats(device)
    c_result = train_model(
        condenser, data, VOCAB_SIZE, device,
        num_steps=NUM_STEPS, lr=3e-4,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        model_name="Condenser",
    )
    c_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"  Condenser peak VRAM: {c_peak:.0f}MB")

    print()
    print("=" * 60)
    print("TRAINING TRANSFORMER")
    print("=" * 60)
    torch.cuda.reset_peak_memory_stats(device)
    t_result = train_model(
        transformer, data, VOCAB_SIZE, device,
        num_steps=NUM_STEPS, lr=3e-4,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        model_name="Transformer",
    )
    t_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"  Transformer peak VRAM: {t_peak:.0f}MB")

    print()
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Condenser':<18} {'Transformer':<18}")
    print("-" * 61)
    print(f"{'Params':<25} {c_params:<18,} {t_params:<18,}")
    print(f"{'Peak VRAM':<25} {c_peak:<18.0f}MB {t_peak:<18.0f}MB")
    print(f"{'Compression':<25} {t_params/c_params:<18.2f}x {'1x':<18}")
    print(f"{'Best PPL':<25} {c_result['best_ppl']:<18.2f} {t_result['best_ppl']:<18.2f}")
    print(f"{'Final Loss':<25} {c_result['final_loss']:<18.4f} {t_result['final_loss']:<18.4f}")
    print(f"{'Avg step ms':<25} {c_result['avg_step_ms']:<18.1f} {t_result['avg_step_ms']:<18.1f}")
    print(f"{'Avg VRAM (last 100)':<25} {c_result['avg_vram']:<18.0f}MB {t_result['avg_vram']:<18.0f}MB")

    vram_ratio = t_peak / c_peak if c_peak > 0 else float('inf')
    print(f"{'VRAM savings':<25} {'':<18} {vram_ratio:.2f}x less memory")
