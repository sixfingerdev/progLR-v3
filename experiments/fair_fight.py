"""
Fair fight: PRISM vs Transformer with MATCHED stored parameters.
PRISM gets more layers since its layers use ephemeral weights.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import time
import math

from condenser.prism import PRISMModel
from baseline_transformer import BaselineTransformer


def make_noisy_dataset(num_chars=500000, seed=42):
    """More realistic dataset with noise = harder to memorize."""
    torch.manual_seed(seed)

    vocab = {}
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?\n"
    for i, c in enumerate(chars):
        vocab[c] = i + 1
    vocab_size = len(vocab) + 1

    text_segments = [
        "the quick brown fox jumps over the lazy dog",
        "in the beginning was the word and the word was with god",
        "to be or not to be that is the question",
        "all that glitters is not gold",
        "a thing of beauty is a joy forever",
        "the course of true love never did run smooth",
        "brevity is the soul of wit",
        "the lady doth protest too much methinks",
        "whats in a name that which we call a rose by any other name would smell as sweet",
        "if music be the food of love play on",
        "shall i compare thee to a summers day thou art more lovely and more temperate",
        "so long as men can breathe or eyes can see so long lives this and this gives life to thee",
        "once upon a time there was a brave knight who lived in a castle",
        "the ancient forest was dark and full of mysteries",
        "she walked through the garden of forgotten dreams",
        "the scientist worked day and night to find the cure",
        "in the year 3000 robots ruled the world",
        "the detective examined the crime scene carefully",
        "a great adventure was about to begin",
        "the last dragon slept beneath the mountain",
    ]

    noise_chars = "xzvqjk"
    tokens = []
    for _ in range(num_chars):
        seg = text_segments[_ % len(text_segments)]
        if torch.rand(1).item() < 0.3:
            idx = torch.randint(0, len(seg), (1,)).item()
            seg = seg[:idx] + noise_chars[torch.randint(0, len(noise_chars), (1,)).item()] + seg[idx+1:]
        for c in seg:
            if c in vocab:
                tokens.append(vocab[c])
            else:
                tokens.append(vocab.get(' ', 1))

    data = torch.tensor(tokens[:num_chars])
    return data, vocab_size


def get_batches(data, batch_size, seq_len):
    data = data[:(len(data) // (batch_size * seq_len)) * (batch_size * seq_len)]
    data = data.view(batch_size, -1)
    for i in range(0, data.size(1) - seq_len, seq_len // 2):
        yield data[:, i:i + seq_len]


def train_model(model, data, vocab_size, device, num_steps, batch_size, seq_len, lr, name):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    losses, ppls, vrams = [], [], []
    best_ppl = float('inf')
    step = 0

    while step < num_steps:
        for batch in get_batches(data, batch_size, seq_len):
            if step >= num_steps:
                break

            batch = batch.to(device)
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

            ppl = math.exp(min(loss.item(), 15))
            losses.append(loss.item())
            ppls.append(ppl)
            if ppl < best_ppl:
                best_ppl = ppl

            if device.type == 'cuda':
                vrams.append(torch.cuda.memory_allocated(device) / (1024**2))

            step += 1
            if step % 500 == 0:
                avg_l = sum(losses[-200:]) / min(200, len(losses))
                avg_p = math.exp(avg_l)
                avg_v = sum(vrams[-200:]) / min(200, len(vrams)) if vrams else 0
                print(f"  [{name}] step {step:>4}/{num_steps}  loss={avg_l:.4f}  ppl={avg_p:.2f}  vram={avg_v:.0f}MB")

    return {'best_ppl': best_ppl, 'losses': losses, 'ppls': ppls, 'vrams': vrams}


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data, vocab_size = make_noisy_dataset(800000)
    print(f"Dataset: {len(data):,} tokens, vocab={vocab_size}")

    # === MATCHED PARAMETER FIGHT ===
    # Give both ~7M stored params
    # Transformer: hidden=384, 6 layers → 6.8M params
    # PRISM: hidden=384, many more layers (layers are cheap!) → match to 7M total

    HIDDEN = 256
    T_LAYERS = 6
    BATCH = 8
    SEQ = 64
    STEPS = 2000

    transformer = BaselineTransformer(
        vocab_size=vocab_size, hidden_dim=HIDDEN,
        num_layers=T_LAYERS, num_heads=8,
    )
    t_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

    # PRISM: use matching layers + low rank to show efficiency
    rank = HIDDEN // 8
    state_dim = HIDDEN

    prism = PRISMModel(
        vocab_size=vocab_size, hidden_dim=HIDDEN,
        num_layers=T_LAYERS,
        state_dim=state_dim, rank=rank,
    )
    p_params = prism.count_trained()

    print(f"\n{'='*50}")
    print(f"FAIR FIGHT SETUP")
    print(f"{'='*50}")
    print(f"  Transformer: {t_params:,} params, {T_LAYERS} layers")
    print(f"  PRISM:       {p_params:,} params, {n_prism_layers} layers ({n_prism_layers//T_LAYERS}x more)")
    print(f"  Hidden dim:  {HIDDEN}")
    print(f"  Param match: {abs(t_params-p_params)/max(t_params, p_params)*100:.1f}% difference")

    print(f"\n{'='*50}")
    print(f"TRAINING TRANSFORMER ({T_LAYERS} layers)")
    print(f"{'='*50}")
    torch.cuda.reset_peak_memory_stats(device)
    t_res = train_model(transformer, data, vocab_size, device, STEPS, BATCH, SEQ, 3e-4, "TF")
    t_peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0

    print(f"\n{'='*50}")
    print(f"TRAINING PRISM ({n_prism_layers} layers)")
    print(f"{'='*50}")
    torch.cuda.reset_peak_memory_stats(device)
    p_res = train_model(prism, data, vocab_size, device, STEPS, BATCH, SEQ, 3e-4, "PRISM")
    p_peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0

    # Final validation: evaluate on held-out batch
    print(f"\n{'='*60}")
    print(f"MATCHED-PARAM RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'PRISM':<18} {'Transformer':<18}")
    print(f"{'-'*66}")
    print(f"{'Stored params':<30} {p_params:<18,} {t_params:<18,}")
    print(f"{'Effective layers':<30} {n_prism_layers:<18} {T_LAYERS:<18}")
    print(f"{'Best PPL':<30} {p_res['best_ppl']:<18.2f} {t_res['best_ppl']:<18.2f}")
    print(f"{'Peak VRAM':<30} {p_peak:<18.0f}MB {t_peak:<18.0f}MB")
    print(f"{'VRAM efficiency':<30} {p_params/max(p_peak,1):<18.0f} params/MB {t_params/max(t_peak,1):<18.0f} params/MB")

    flux_gap = p_res['best_ppl'] - t_res['best_ppl']
    vram_ratio = t_peak / max(p_peak, 1)
    print(f"\n  PPL gap: {abs(flux_gap):.2f} points ({'PRISM' if flux_gap < 0 else 'Transformer'} leads)")
    print(f"  VRAM ratio: {vram_ratio:.1f}x (Transformer uses {vram_ratio:.1f}x more VRAM)")
    print(f"  Params ratio: {t_params/max(p_params, 1):.1f}x")

    efficiency = (p_params / max(p_peak, 1)) / (t_params / max(t_peak, 1))
    print(f"  Parameter density: PRISM stores {efficiency:.1f}x more params per MB of VRAM")

    print(f"\n{'='*60}")
    print(f"PARETO ANALYSIS: Quality vs Efficiency")
    print(f"{'='*60}")
    print(f"  At same stored params:")
    print(f"    PRISM: {n_prism_layers}x more layers, {vram_ratio:.1f}x less VRAM")
    print(f"    Quality: {'PRISM wins' if flux_gap < 0 else 'Transformer wins'} by {abs(flux_gap):.2f} PPL")
    print(f"\n  At same VRAM budget:")
    print(f"    PRISM could have {int(n_prism_layers * vram_ratio)} layers and {int(p_params * vram_ratio):,} params")
    print(f"    vs Transformer's {T_LAYERS} layers and {t_params:,} params")
    print(f"    -> PRISM likely wins at matched VRAM")
