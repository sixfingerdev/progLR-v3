"""
V3-PRISM: OOD benchmark on arithmetic (a+b mod 100).
PRISM layers now have full attention+FFN with ephemeral weights.
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.v3_prism import V3PRISM
from condenser.baseline_transformer import BaselineTransformer
from condenser.v3_all import V3AllTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB', flush=True)

NUM = 100
PLUS = NUM; EQUALS = NUM + 1; VOCAB = NUM + 2

pairs = [(a, b) for a in range(NUM) for b in range(NUM)]
c_sum = [(a + b) % NUM for a, b in pairs]

import random
random.seed(42)
indices = list(range(len(pairs)))
random.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx = indices[:split]; test_idx = indices[split:]

def make_data(idx_list):
    seqs = []
    for idx in idx_list:
        a, b = pairs[idx]; c = c_sum[idx]
        seqs.append([a, PLUS, b, EQUALS, c])
    return torch.tensor(seqs, dtype=torch.long, device=device)

train_data = make_data(train_idx); test_data = make_data(test_idx)
print(f'Pairs: {len(pairs)}, Train: {len(train_idx)}, Test: {len(test_idx)}', flush=True)

def evaluate(model, data):
    model.eval(); total = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], 256):
            b = data[i:i+256]
            logits = model(b[:, :4])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1))
            total += loss.item() * b.shape[0]; n += b.shape[0]
    return total / n

def accuracy(model, data):
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], 256):
            b = data[i:i+256]
            logits = model(b[:, :4])
            preds = logits.argmax(dim=-1)
            correct += (preds[:, -1] == b[:, -1]).sum().item()
            total += b.shape[0]
    return correct / total * 100

def trained_params(model):
    if hasattr(model, 'count_trained'): return model.count_trained()
    if hasattr(model, 'count_params'): return model.count_params()
    return sum(p.numel() for p in model.parameters())

def eff_params(model):
    if hasattr(model, 'effective_params'): return model.effective_params()
    return model.count_params()

# === TEST at small scale ===
D, L = 128, 4
B, STEPS = 256, 10000

configs = [
    ('Std Transformer', BaselineTransformer(VOCAB, D, L, 4)),
    ('V3-PRISM 2A+2P r=16', V3PRISM(VOCAB, D, L, num_anchor=2, rank=16, embed_rank=8)),
    ('V3-PRISM 0A+4P r=16', V3PRISM(VOCAB, D, L, num_anchor=0, rank=16, embed_rank=8)),
    ('V3-All 2Std+2LR r=8', V3AllTransformer(VOCAB, D, L, 4, embed_rank=8, hidden_rank=8, num_std_layers=2)),
]

print(f'\n{"="*70}', flush=True)
print(f'  OOD BENCHMARK: V3-PRISM (with attention) vs V3-All vs Transformer', flush=True)
print(f'{"="*70}', flush=True)
for name, model in configs:
    print(f'  {name:25}: {trained_params(model):>7,} trained | eff={eff_params(model)/1e6:.1f}M', flush=True)

print(f'\n  {"Model":25} {"Step":>6} {"Loss":>8} {"OOD%":>6} {"PPL":>7}', flush=True)
print(f'  {"-"*55}', flush=True)

results = {}
for name, model in configs:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()

    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, train_data.shape[0], (B,))
        batch = train_data[idx]
        loss = F.cross_entropy(model(batch[:, :4]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        if step % 2000 == 0:
            acc = accuracy(model, test_data[:500])
            tst = evaluate(model, test_data[:500])
            elapsed = time.time() - t0
            print(f'  {name:25} {step:6d} {loss.item():8.4f} {acc:5.1f}% {math.exp(tst):6.1f} [{elapsed:.0f}s]', flush=True)

    test_acc = accuracy(model, test_data)
    test_loss = evaluate(model, test_data)
    vram = torch.cuda.max_memory_allocated(device) / 1e6
    results[name] = {'acc': test_acc, 'loss': test_loss, 'ppl': math.exp(test_loss), 'vram': vram}
    tp = trained_params(model)
    ep = eff_params(model)
    suffix = ''
    if hasattr(model, 'memory_report'):
        r = model.memory_report()
        suffix = f' | eff={r["effective_b"]:.2f}B | comp={r["compression"]:.0f}x'

    print(f'  >>> [{name}] OOD={test_acc:.1f}% | VRAM={vram:.0f}MB | '
          f'trained={tp/1e6:.2f}M | eff_total={ep/1e6:.1f}M{suffix}', flush=True)

# === VRAM SCALE to 100B ===
print(f'\n{"="*80}', flush=True)
print(f'  VRAM SCALE: V3-PRISM to 100B', flush=True)
print(f'{"="*80}', flush=True)
print(f'  {"D":>6} {"L":>4} {"A":>3} {"Eff(B)":>9} {"Trained":>10} {"VRAM":>8} {"Fits?":>6}', flush=True)
print(f'  {"-"*55}', flush=True)

scale_configs = [
    (4096,  24, 0, 500_000, 16, 8),
    (4096,  48, 0, 500_000, 16, 8),
    (8192,  32, 0, 500_000, 16, 8),
    (16384, 48, 0, 500_000, 16, 8),
    (16384, 96, 0, 500_000, 16, 8),
    (24576, 64, 0, 500_000, 16, 8),
    (32768, 64, 0, 500_000, 16, 8),
]

for D, L, A, V, R, ER in scale_configs:
    m = V3PRISM(V, D, L, num_anchor=A, rank=R, embed_rank=ER)
    r = m.memory_report()
    print(f'  {D:>6} {L:>4} {A:>3} {r["effective_b"]:>7.1f}B {r["trained_m"]:>8.1f}M '
          f'{r["total_gb"]:>6.1f}GB {"YES" if r["fits_6gb"] else "NO":>6}', flush=True)

print(f'\n{"="*50}', flush=True)
print(f'  RESULTS', flush=True)
print(f'{"="*50}', flush=True)
for name, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
    print(f'  {name:25}: {r["acc"]:.1f}% OOD | {r["ppl"]:.1f} PPL | {r["vram"]:.0f}MB', flush=True)
print(f'  Random: {100/VOCAB:.1f}%', flush=True)
