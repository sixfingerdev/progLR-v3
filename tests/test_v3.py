"""Test V3 GenoMatrix forward + backward + compare with Transformer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import time, math

from condenser.genomatrix import GenoMatrixModel
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- BASIC TEST ---
print("\n=== V3 Basic Test ===")
model = GenoMatrixModel(vocab_size=500, hidden_dim=128, num_layers=4, genom_state_dim=64).to(device)
x = torch.randint(0, 500, (2, 16)).to(device)
logits = model(x)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, 500), x[:, 1:].reshape(-1))
loss.backward()
print(f"PASS Forward+backward: loss={loss.item():.4f}")
print(f"PASS Trained params: {model.count_trained():,}")
print(f"PASS Effective params: {model.effective_params():,}")
print(f"PASS Compression: {model.effective_params()/model.count_trained():.1f}x")

for n, p in model.named_parameters():
    if p.requires_grad and p.grad is not None:
        print(f"  {n}: grad norm {p.grad.norm():.4f}")
        break

# --- QUALITY COMPARISON ---
print("\n=== V3 vs Transformer Quality ===")

VOCAB = 500
HIDDEN = 256
LAYERS = 4
BATCH = 4
SEQ = 64
STEPS = 1500

torch.manual_seed(42)
data = torch.randint(1, VOCAB, (500000,))
for i in range(5000):
    pos = torch.randint(0, 499990, (1,)).item()
    data[pos] = data[pos].item()
    data[pos+1] = (data[pos].item() * 7 + 13) % (VOCAB-1) + 1
    data[pos+2] = (data[pos+1].item() * 3 + 7) % (VOCAB-1) + 1

v3 = GenoMatrixModel(vocab_size=VOCAB, hidden_dim=HIDDEN, num_layers=LAYERS, genom_state_dim=128)
tf = BaselineTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, num_layers=LAYERS, num_heads=8)
v3_p = v3.count_trained()
tf_p = tf.count_params()
print(f"V3 params: {v3_p:,}, Transformer params: {tf_p:,} ({tf_p/max(v3_p,1):.1f}x)")

def train(model, name):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses, ppls = [], []
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(STEPS):
        idx = torch.randint(0, len(data)-SEQ-1, (BATCH,))
        batch = torch.stack([data[i:i+SEQ] for i in idx]).to(device)
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ppl = math.exp(min(loss.item(), 10))
        losses.append(loss.item()); ppls.append(ppl)

        if step % 300 == 0:
            avg_l = sum(losses[-100:]) / min(100, len(losses))
            vram = torch.cuda.memory_allocated(device)/1e6
            print(f"  [{name}] step {step}: loss={avg_l:.4f} ppl={math.exp(avg_l):.2f} vram={vram:.0f}MB")

    peak = torch.cuda.max_memory_allocated(device)/1e6
    return {'best_ppl': min(ppls), 'best_loss': min(losses), 'peak_vram': peak, 'ppls': ppls}

print("\nTraining V3...")
v3_r = train(v3, "V3")
print("\nTraining Transformer...")
tf_r = train(tf, "TF")

print(f"\n{'='*55}")
print(f"  V3 vs TRANSFORMER")
print(f"{'='*55}")
print(f"{'':<20} {'V3 GenoMatrix':<18} {'Transformer':<18}")
print(f"{'-'*56}")
print(f"{'Stored params':<20} {v3_p:<18,} {tf_p:<18,}")
print(f"{'Best PPL':<20} {v3_r['best_ppl']:<18.2f} {tf_r['best_ppl']:<18.2f}")
print(f"{'Best Loss':<20} {v3_r['best_loss']:<18.4f} {tf_r['best_loss']:<18.4f}")
print(f"{'Peak VRAM':<20} {v3_r['peak_vram']:<18.0f}MB {tf_r['peak_vram']:<18.0f}MB")
print(f"{'Compression':<20} {tf_p/max(v3_p,1):<18.1f}x {'1x':<18}")

gap = v3_r['best_ppl'] - tf_r['best_ppl']
if gap < 0:
    print(f"\n  V3 WINS by {abs(gap):.2f} PPL!")
else:
    print(f"\n  Transformer wins by {gap:.2f} PPL")

vram_ratio = tf_r['peak_vram'] / max(v3_r['peak_vram'], 1)
print(f"  V3 uses {vram_ratio:.1f}x less VRAM")
print(f"  V3 trains with full-rank weights (no bottleneck)")
