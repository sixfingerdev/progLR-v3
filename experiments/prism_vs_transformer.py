import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import time
import math

from condenser.prism import PRISMModel
from baseline_transformer import BaselineTransformer


def generate_dataset(num_tokens=200000, vocab_size=500, seed=42):
    torch.manual_seed(seed)
    data = torch.randint(1, vocab_size, (num_tokens,))

    patterns = [
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 20, 30, 40, 50],
        [100, 200, 300],
        [50, 40, 30, 20, 10],
    ]
    for i in range(500):
        pat = patterns[i % len(patterns)]
        pos = torch.randint(0, num_tokens - len(pat) - 1, (1,)).item()
        data[pos:pos+len(pat)] = torch.tensor(pat, dtype=torch.long)

    return data


def get_batches(data, batch_size, seq_len):
    data = data[: (len(data) // (batch_size * seq_len)) * (batch_size * seq_len)]
    data = data.view(batch_size, -1)
    for i in range(0, data.size(1) - seq_len, seq_len):
        yield data[:, i:i + seq_len]


def train_model(model, data, vocab_size, device, num_steps=500,
                lr=3e-4, batch_size=4, seq_len=128, name="Model"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    losses = []
    ppls = []
    vrams = []
    step = 0
    best_ppl = float('inf')

    while step < num_steps:
        for batch in get_batches(data, batch_size, seq_len):
            if step >= num_steps:
                break
            batch = batch.to(device)

            t0 = time.time()
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            t1 = time.time()

            ppl = math.exp(loss.item())
            losses.append(loss.item())
            ppls.append(ppl)

            if ppl < best_ppl:
                best_ppl = ppl

            if device.type == 'cuda':
                vrams.append(torch.cuda.memory_allocated(device) / (1024**2))

            step += 1
            if step % 200 == 0:
                avg = sum(losses[-100:]) / min(100, len(losses))
                avg_vram = sum(vrams[-100:]) / min(100, len(vrams)) if vrams else 0
                print(f"  {name} step {step:>4}: loss={avg:.4f} ppl={math.exp(avg):.1f} "
                      f"vram={avg_vram:.0f}MB {t1-t0:.3f}s/step")
    return {
        'losses': losses,
        'ppls': ppls,
        'vrams': vrams,
        'best_ppl': best_ppl,
        'final_loss': losses[-1] if losses else 0,
    }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    VOCAB = 500
    HIDDEN = 256
    LAYERS = 6
    SEQLEN = 128
    BATCH = 4
    STEPS = 2000

    data = generate_dataset(num_tokens=500000, vocab_size=VOCAB)

    print(f"\nConfig: hidden={HIDDEN}, layers={LAYERS}, seq={SEQLEN}, "
          f"batch={BATCH}, steps={STEPS}")
    print()

    prism = PRISMModel(
        vocab_size=VOCAB,
        hidden_dim=HIDDEN,
        num_layers=LAYERS,
        state_dim=128,
        rank=16,
    )
    transformer = BaselineTransformer(
        vocab_size=VOCAB,
        hidden_dim=HIDDEN,
        num_layers=LAYERS,
        num_heads=8,
    )

    p_params = prism.count_trained()
    t_params = transformer.count_params()
    print(f"PRISM trained params: {p_params:,} ({p_params*2/1e6:.1f}MB fp16)")
    print(f"Transformer params: {t_params:,} ({t_params*2/1e6:.1f}MB fp16)")
    print(f"Compression: {t_params/p_params:.1f}x")
    print()

    torch.cuda.reset_peak_memory_stats(device)

    print("=" * 50)
    print("TRAINING PRISM (ephemeral weights)")
    print("=" * 50)
    torch.cuda.reset_peak_memory_stats(device)
    p_result = train_model(prism, data, VOCAB, device,
                           num_steps=STEPS, batch_size=BATCH,
                           seq_len=SEQLEN, name="PRISM")
    p_peak = torch.cuda.max_memory_allocated(device) / (1024**2)

    print(f"PRISM peak VRAM: {p_peak:.0f}MB, best PPL: {p_result['best_ppl']:.1f}")
    print()

    print("=" * 50)
    print("TRAINING TRANSFORMER")
    print("=" * 50)
    torch.cuda.reset_peak_memory_stats(device)
    t_result = train_model(transformer, data, VOCAB, device,
                           num_steps=STEPS, batch_size=BATCH,
                           seq_len=SEQLEN, name="Transformer")
    t_peak = torch.cuda.max_memory_allocated(device) / (1024**2)

    print(f"Transformer peak VRAM: {t_peak:.0f}MB, best PPL: {t_result['best_ppl']:.1f}")
    print()

    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'PRISM':<18} {'Transformer':<18}")
    print("-" * 61)
    print(f"{'Params':<25} {p_params:<18,} {t_params:<18,}")
    print(f"{'Peak VRAM':<25} {p_peak:<18.0f}MB {t_peak:<18.0f}MB")
    print(f"{'VRAM Ratio':<25} 1x{'':<18} {t_peak/max(p_peak,1):<18.1f}x")
    print(f"{'Compression':<25} {t_params/p_params:<18.1f}x 1x")
    print(f"{'Best PPL':<25} {p_result['best_ppl']:<18.1f} {t_result['best_ppl']:<18.1f}")
    print(f"{'Final Loss':<25} {p_result['final_loss']:<18.4f} {t_result['final_loss']:<18.4f}")

    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"PRISM uses {t_params/p_params:.1f}x fewer params and {(t_peak/max(p_peak,1)):.1f}x less VRAM")
    print(f"PRISM PPL gap: {t_result['best_ppl'] - p_result['best_ppl']:.1f} points")
    print("(PRISM generates weights on-the-fly, storing only tiny Genom)")
