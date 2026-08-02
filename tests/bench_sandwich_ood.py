"""
Sandwich (interleaved) ProgLR OOD benchmark on a+b mod 100.
Protocol (AGENTS.md): train a in [0,50) -> test a in [50,100)  (extrapolation).
Also reports the 80/20 random-split protocol used in bench_v3_all.py for
comparability with published V3-All numbers.

Designs at L=12 (scaled copy of the 32-layer sandwich plan
[Std, Std, LR, LR, LR, LR, Std, LR, LR, LR, LR, Std]):
  1. Std Transformer (12 std)
  2. Stacked 3Std+9LR r=16  (classic: std at bottom, LR on top)
  3. Sandwich 4Std+8LR r=16 (user design)
  4. Sandwich 4Std+8LR r=8
  5. All-LR 12 r=16         (no anchors, control)
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.v3_all import V3AllTransformer
from condenser.baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB', flush=True)

NUM = 100
PLUS = NUM
EQUALS = NUM + 1
VOCAB = NUM + 2
RANDOM_CHANCE = 100 / VOCAB

def make_data(pairs):
    seqs = []
    for a, b in pairs:
        seqs.append([a, PLUS, b, EQUALS, (a + b) % NUM])
    return torch.tensor(seqs, dtype=torch.long, device=device)

# --- Protocol 1: extrapolation OOD (train a<50 -> test a>=50) ---
train_pairs = [(a, b) for a in range(50) for b in range(NUM)]
test_pairs = [(a, b) for a in range(50, NUM) for b in range(NUM)]
train_o = make_data(train_pairs)
test_o = make_data(test_pairs)
print(f'OOD protocol: train a in [0,50) x b in [0,100) = {len(train_o)}, '
      f'test a in [50,100) = {len(test_o)}', flush=True)

# --- Protocol 2: random 80/20 (bench_v3_all compatibility) ---
import random
random.seed(42)
all_pairs = [(a, b) for a in range(NUM) for b in range(NUM)]
indices = list(range(len(all_pairs)))
random.shuffle(indices)
split = int(len(indices) * 0.8)
train_r = make_data([all_pairs[i] for i in indices[:split]])
test_r = make_data([all_pairs[i] for i in indices[split:]])

def evaluate(model, data):
    model.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], 256):
            b = data[i:i+256]
            logits = model(b[:, :4])
            total += F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1)).item() * b.shape[0]
            n += b.shape[0]
    return total / n

def accuracy(model, data):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], 256):
            b = data[i:i+256]
            preds = model(b[:, :4]).argmax(dim=-1)
            correct += (preds[:, -1] == b[:, -1]).sum().item()
            total += b.shape[0]
    return correct / total * 100

SANDWICH = ['std', 'std'] + ['lr'] * 4 + ['std'] + ['lr'] * 4 + ['std']

configs = [
    ('Std Transformer L=12',
     BaselineTransformer(VOCAB, 1024, 12, 16)),
    ('Stacked 3Std+9LR r=16',
     V3AllTransformer(VOCAB, 1024, 12, 16, embed_rank=8, hidden_rank=16, num_std_layers=3)),
    ('Sandwich 4Std+8LR r=16',
     V3AllTransformer(VOCAB, 1024, 12, 16, embed_rank=8, hidden_rank=16, layer_pattern=SANDWICH)),
    ('Sandwich 4Std+8LR r=8',
     V3AllTransformer(VOCAB, 1024, 12, 16, embed_rank=8, hidden_rank=8, layer_pattern=SANDWICH)),
    ('All-LR 12 r=16 (no anchors)',
     V3AllTransformer(VOCAB, 1024, 12, 16, embed_rank=8, hidden_rank=16, num_std_layers=0)),
]

D, B = 1024, 64
STEPS = int(os.environ.get('SANDWICH_STEPS', '6000'))
SEED = int(os.environ.get('SANDWICH_SEED', '0'))
if SEED:
    torch.manual_seed(SEED)
SELECT = [s.strip().lower() for s in os.environ.get('SANDWICH_SELECT', '').split(',') if s.strip()]
if SELECT:
    configs = [c for c in configs if any(s in c[0].lower() for s in SELECT)]
print(f'\nD={D}, L=12, B={B}, steps={STEPS}, seed={SEED or "default"}, configs={len(configs)}', flush=True)

results = {}
for name, model in configs:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()
    best_acc = 0.0
    print(f'\n--- {name} ({model.count_params():,} params) ---', flush=True)
    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, train_o.shape[0], (B,))
        batch = train_o[idx]
        loss = F.cross_entropy(model(batch[:, :4]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0 and step > 0:
            acc = accuracy(model, test_o[:1000])
            best_acc = max(best_acc, acc)
            print(f'  st={step:5d} | loss={loss.item():.4f} | OOD={acc:.1f}% | best={best_acc:.1f}%', flush=True)
    dt = time.time() - t0
    acc_o = accuracy(model, test_o)
    acc_r = accuracy(model, test_r)
    ppl_o = math.exp(evaluate(model, test_o))
    results[name] = {'acc_o': acc_o, 'acc_r': acc_r, 'ppl_o': ppl_o, 'params': model.count_params(),
                     'seconds': dt}
    print(f'  RESULT {name}: OOD(extrap)={acc_o:.1f}% | 80/20={acc_r:.1f}% | ppl={ppl_o:.1f} | {dt:.0f}s', flush=True)

print(f'\n{"="*90}', flush=True)
print(f'  {"Model":34} {"Params":>9} {"OOD a>=50":>10} {"80/20":>7} {"PPL":>6} {"sec":>6}', flush=True)
print(f'  {"-"*80}', flush=True)
for name, r in sorted(results.items(), key=lambda x: -x[1]['acc_o']):
    print(f'  {name:34} {r["params"]:>9,} {r["acc_o"]:>9.1f}% {r["acc_r"]:>6.1f}% {r["ppl_o"]:>6.1f} {r["seconds"]:>6.0f}', flush=True)
print(f'  Random chance: {RANDOM_CHANCE:.1f}%', flush=True)
