"""Quick honest quality benchmark - small & fast."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import time, math

from condenser.prism import PRISMModel
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Tiny real-like dataset
data = torch.randint(1, 300, (100000,))
# Add some patterns (bi-gram, tri-gram statistics)
for i in range(10000):
    pos = torch.randint(0, 99900, (1,)).item()
    data[pos] = data[pos].item()
    data[pos+1] = (data[pos].item() * 7 + 13) % 299 + 1

VOCAB = 300
HIDDEN = 192
LAYERS = 4
BATCH = 4
SEQ = 64
STEPS = 1000

prism = PRISMModel(vocab_size=VOCAB, hidden_dim=HIDDEN, num_layers=LAYERS, state_dim=128, rank=HIDDEN//4)
transformer = BaselineTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, num_layers=LAYERS, num_heads=6)

p_p = prism.count_trained()
t_p = transformer.count_params()
print(f"PRISM params: {p_p:,}, Transformer params: {t_p:,} ({t_p/max(p_p,1):.1f}x)")

def run(model, name):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses, ppls, vrams = [], [], []
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(STEPS):
        idx = torch.randint(0, len(data)-SEQ-1, (BATCH,))
        batch = torch.stack([data[i:i+SEQ] for i in idx]).to(device)
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        ppl = math.exp(min(loss.item(), 10))
        losses.append(loss.item()); ppls.append(ppl)
        vrams.append(torch.cuda.memory_allocated(device)/1e6)

        if step % 250 == 0:
            print(f"  [{name}] step {step}: loss={loss.item():.4f} ppl={ppl:.2f} vram={vrams[-1]:.0f}MB")

    peak = torch.cuda.max_memory_allocated(device)/1e6
    return {'best_ppl': min(ppls), 'peak_vram': peak, 'losses': losses, 'ppls': ppls, 'vrams': vrams}

print("\n--- PRISM ---")
p_r = run(prism, "PRISM")
print(f"\n--- TRANSFORMER ---")
t_r = run(transformer, "TRANSFORMER")

print(f"\n{'='*55}")
print(f"  FINAL RESULTS")
print(f"{'='*55}")
print(f"{'':<20} {'PRISM':<15} {'Transformer':<15}")
print(f"{'Stored params':<20} {p_p:<15,} {t_p:<15,}")
print(f"{'Best PPL':<20} {p_r['best_ppl']:<15.2f} {t_r['best_ppl']:<15.2f}")
print(f"{'Peak VRAM':<20} {p_r['peak_vram']:<15.0f}MB {t_r['peak_vram']:<15.0f}MB")
print(f"{'Par/VRAM ratio':<20} {p_p/max(p_r['peak_vram'],1):<15.0f} {t_p/max(t_r['peak_vram'],1):<15.0f}")

gap = p_r['best_ppl'] - t_r['best_ppl']
print(f"\n  PPL gap: {gap:.2f} ({'PRISM wins' if gap < 0 else 'Transformer wins'})")
print(f"  VRAM ratio: {t_r['peak_vram']/max(p_r['peak_vram'],1):.1f}x")
print(f"  Param ratio: {t_p/max(p_p,1):.1f}x")
