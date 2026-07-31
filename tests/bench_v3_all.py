"""
V3-All: Final comprehensive benchmark.
Tests architecture variants at multiple scales.
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

pairs = [(a, b) for a in range(NUM) for b in range(NUM)]
c_sum = [(a + b) % NUM for a, b in pairs]

import random
random.seed(42)
indices = list(range(len(pairs)))
random.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx = indices[:split]
test_idx = indices[split:]

def make_data(idx_list):
    seqs = []
    for idx in idx_list:
        a, b = pairs[idx]
        c = c_sum[idx]
        seqs.append([a, PLUS, b, EQUALS, c])
    return torch.tensor(seqs, dtype=torch.long, device=device)

train_data = make_data(train_idx)
test_data = make_data(test_idx)
print(f'Pairs: {len(pairs)}, Train: {len(train_idx)}, Test: {len(test_idx)}', flush=True)

def evaluate(model, data):
    model.eval()
    total = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], 256):
            b = data[i:i+256]
            logits = model(b[:, :4])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1))
            total += loss.item() * b.shape[0]; n += b.shape[0]
    return total / n

def accuracy(model, data):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], 256):
            b = data[i:i+256]
            logits = model(b[:, :4])
            preds = logits.argmax(dim=-1)
            correct += (preds[:, -1] == b[:, -1]).sum().item()
            total += b.shape[0]
    return correct / total * 100

# === BENCHMARK: V3-All vs Baseline at D=128, L=4 ===
D, L, HEADS = 128, 4, 4
B, STEPS = 64, 5000

configs = [
    ('V3-All 2Std+2LR r=8', V3AllTransformer(VOCAB, D, L, HEADS, embed_rank=8, hidden_rank=8, num_std_layers=2)),
    ('V3-All 1Std+3LR r=16', V3AllTransformer(VOCAB, D, L, HEADS, embed_rank=8, hidden_rank=16, num_std_layers=1)),
    ('V3-All 0Std+4LR r=32', V3AllTransformer(VOCAB, D, L, HEADS, embed_rank=8, hidden_rank=32, num_std_layers=0)),
    ('Std Transformer', BaselineTransformer(VOCAB, D, L, HEADS)),
]

print(f'\n{"="*80}', flush=True)
print(f'  BENCHMARK: V3-All vs Standard Transformer (D={D}, L={L})', flush=True)
print(f'{"="*80}', flush=True)
print(f'  {"Model":30} {"Params":>8}  {"Mem/GB"}', flush=True)
print(f'  {"-"*70}', flush=True)
for name, model in configs:
    est = model.vram_estimate()["total_gb"] if hasattr(model, 'vram_estimate') else 0
    print(f'  {name:30} {model.count_params():>8,}  {est:.2f}', flush=True)

print(f'\n  {"Model":30} {"Step":>5} {"Loss":>8} {"OOD%":>6} {"PPL":>7}', flush=True)
print(f'  {"-"*60}', flush=True)

results = {}
for name, model in configs:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, train_data.shape[0], (B,))
        batch = train_data[idx]
        loss = F.cross_entropy(model(batch[:, :4]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        if step % 1000 == 0:
            acc = accuracy(model, test_data[:500])
            tst = evaluate(model, test_data[:500])
            print(f'  {name:30} {step:5d} {loss.item():8.4f} {acc:5.1f}% {math.exp(tst):6.1f}', flush=True)

    test_loss = evaluate(model, test_data)
    test_acc = accuracy(model, test_data)
    vram = torch.cuda.max_memory_allocated(device) / 1e6
    results[name] = {'loss': test_loss, 'acc': test_acc, 'ppl': math.exp(test_loss), 'vram': vram}
    print(f'  RESULT: {name}: {test_acc:.1f}% OOD, {vram:.0f}MB VRAM', flush=True)

# === VRAM SCALE TEST ===
print(f'\n{"="*80}', flush=True)
print(f'  VRAM SCALE TEST: How big can we go in 6GB?', flush=True)
print(f'{"="*80}', flush=True)

scale_configs = [
    ('V3-All 2Std+2LR r=8, D=2048', V3AllTransformer(VOCAB, 2048, 4, 16, embed_rank=8, hidden_rank=8, num_std_layers=2)),
    ('V3-All 1Std+3LR r=16, D=2048', V3AllTransformer(VOCAB, 2048, 4, 16, embed_rank=8, hidden_rank=16, num_std_layers=1)),
    ('V3-All 1Std+7LR r=16, D=4096', V3AllTransformer(VOCAB, 4096, 8, 16, embed_rank=8, hidden_rank=16, num_std_layers=1)),
    ('Std Transformer D=2048 L=4', BaselineTransformer(VOCAB, 2048, 4, 16)),
]

print(f'\n  {"Model":35} {"Params":>10} {"Model":>6} {"Optim":>6} {"Act":>6} {"Total":>6} {"Fits?":>6}', flush=True)
print(f'  {"-"*80}', flush=True)
for name, model in scale_configs:
    if hasattr(model, 'vram_estimate'):
        r = model.vram_estimate()
    else:
        p = model.count_params()
        r = {'model_mb': p*2/1e6, 'optim_mb': p*8/1e6, 'act_mb': 2048*512*4*2/1e6,
             'total_gb': (p*2/1e6 + p*8/1e6 + 2048*512*4*2/1e6)/1000, 'fits_6gb': False}
    print(f'  {name:35} {model.count_params():>10,} {r["model_mb"]:>5.0f}MB {r["optim_mb"]:>5.0f}MB '
          f'{r["act_mb"]:>5.0f}MB {r["total_gb"]:>5.1f}GB {"YES" if r.get("fits_6gb",False) else "NO":>6}', flush=True)

# === FINAL TABLE ===
print(f'\n{"="*80}', flush=True)
print(f'  FINAL RESULTS', flush=True)
print(f'{"="*80}', flush=True)
print(f'  {"Model":30} {"Params":>8} {"OOD%":>6} {"PPL":>7} {"VRAM":>7}', flush=True)
print(f'  {"-"*60}', flush=True)
for name, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
    m = [m for n, m in configs if n == name][0]
    print(f'  {name:30} {m.count_params():>8,} {r["acc"]:5.1f}% {r["ppl"]:6.1f} {r["vram"]:6.0f}MB', flush=True)

print(f'\n  Random chance: {100/VOCAB:.1f}%', flush=True)
