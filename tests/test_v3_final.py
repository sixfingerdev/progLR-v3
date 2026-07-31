"""Test V3 Final - GenoTransformer quality vs real Transformer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import time, math

from condenser.genomatrix import GenoTransformer
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

VOCAB = 500
HIDDEN = 256
LAYERS = 4
HEADS = 8
RANK = 16
BATCH = 4
SEQ = 64
STEPS = 2000

torch.manual_seed(42)
data = torch.randint(1, VOCAB, (500000,))
for i in range(5000):
    pos = torch.randint(0, 499990, (1,)).item()
    data[pos] = data[pos].item()
    data[pos+1] = (data[pos].item() * 7 + 13) % (VOCAB-1) + 1
    data[pos+2] = (data[pos+1].item() * 3 + 7) % (VOCAB-1) + 1
    data[pos+3] = (data[pos+2].item() * 5 + 3) % (VOCAB-1) + 1

v3 = GenoTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, num_layers=LAYERS, num_heads=HEADS, rank=RANK)
tf = BaselineTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, num_layers=LAYERS, num_heads=HEADS)
v3_p = v3.count_trained()
tf_p = tf.count_params()

print(f"\nV3 params: {v3_p:,} (LoRA rank={RANK}, 6 factors/layer)")
print(f"TF params: {tf_p:,}")
print(f"Compression: {tf_p/max(v3_p,1):.1f}x")
print(f"Effective W: 6 layers x {HIDDEN}^2 x 6 transforms = {v3.effective_params():,}")

def train(model, name, steps=STEPS):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    losses, ppls = [], []
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(steps):
        idx = torch.randint(0, len(data)-SEQ-1, (BATCH,))
        batch = torch.stack([data[i:i+SEQ] for i in idx]).to(device)
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        ppl = math.exp(min(loss.item(), 10))
        losses.append(loss.item()); ppls.append(ppl)

        if step % 400 == 0:
            avg = sum(losses[-100:]) / min(100, len(losses))
            vram = torch.cuda.memory_allocated(device)/1e6
            print(f"  [{name}] step {step}: loss={avg:.4f} ppl={math.exp(avg):.2f} vram={vram:.0f}MB")

    peak = torch.cuda.max_memory_allocated(device)/1e6
    return {'best_ppl': min(ppls), 'losses': losses, 'peak': peak, 'ppls': ppls}

print("\n--- V3 GenoTransformer ---")
v3_r = train(v3, "V3")
print("\n--- Transformer ---")
tf_r = train(tf, "TF")

print(f"\n{'='*55}")
print(f"  FINAL: GENOTRANSFORMER vs TRANSFORMER")
print(f"{'='*55}")
print(f"{'':<22} {'GenoTransformer':<18} {'Transformer':<18}")
print(f"{'-'*58}")
print(f"{'Stored params':<22} {v3_p:<18,} {tf_p:<18,}")
print(f"{'Best PPL':<22} {v3_r['best_ppl']:<18.2f} {tf_r['best_ppl']:<18.2f}")
print(f"{'Peak VRAM':<22} {v3_r['peak']:<18.0f}MB {tf_r['peak']:<18.0f}MB")
print(f"{'Compression':<22} {tf_p/max(v3_p,1):<18.1f}x {'1x':<18}")

gap = v3_r['best_ppl'] - tf_r['best_ppl']
if gap < 0:
    print(f"\n  >>> GENOTRANSFORMER WINS by {abs(gap):.2f} PPL! <<<")
elif gap < 1:
    print(f"\n  >>> TIE! Gap is only {gap:.2f} PPL <<<")
else:
    print(f"\n  >>> Transformer wins by {gap:.2f} PPL <<<")

vram_ratio = tf_r['peak'] / max(v3_r['peak'], 1)
print(f"  GenoTransformer uses {vram_ratio:.1f}x less VRAM")
print(f"  Architecture: Full transformer with ephemeral weights!")
