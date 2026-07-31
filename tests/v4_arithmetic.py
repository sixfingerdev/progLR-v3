"""V4 Spectral Steering on arithmetic reasoning."""
import sys, os, math, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from condenser.genomatrix_v4 import GenoTransformerV4
from condenser.baseline_transformer import BaselineTransformer
from condenser.genomatrix import GenoTransformer as GenoTransformerV3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

NUM = 100; PLUS = NUM; EQUALS = NUM + 1; VOCAB = NUM + 2
pairs = [(a, b) for a in range(NUM) for b in range(NUM)]
c_sum = [(a + b) % NUM for a, b in pairs]
random.seed(42); indices = list(range(len(pairs))); random.shuffle(indices)
split = int(len(indices) * 0.8)

def make_data(idx_list):
    s = []
    for idx in idx_list:
        a, b = pairs[idx]; c = c_sum[idx]
        s.append([a, PLUS, b, EQUALS, c])
    return torch.tensor(s, dtype=torch.long, device=device)

td = make_data(indices[:split]); vd = make_data(indices[split:])
print(f'Data: train {td.shape[0]}, test {vd.shape[0]}, vocab {VOCAB}')

D, L, HEADS, B, STEPS = 128, 4, 4, 64, 5000

configs = [
    ('V3(r=8)', GenoTransformerV3(VOCAB, D, L, HEADS, rank=8, alpha=1.0)),
    ('V4(r=8)', GenoTransformerV4(VOCAB, D, L, HEADS, rank=8, alpha=1.0)),
    ('V4(r=16)', GenoTransformerV4(VOCAB, D, L, HEADS, rank=16, alpha=1.0)),
    ('Transformer', BaselineTransformer(VOCAB, D, L, HEADS)),
]

def count_params(m):
    return m.count_trained() if hasattr(m, 'count_trained') else m.count_params()

def evaluate(model, data):
    model.eval(); t = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]; logits = model(b[:, :4])
            t += F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1)).item() * b.shape[0]; n += b.shape[0]
    return t / n

def accuracy(model, data):
    model.eval(); c = 0; t = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]; preds = model(b[:, :4]).argmax(dim=-1)
            c += (preds[:, -1] == b[:, -1]).sum().item(); t += b.shape[0]
    return c / t * 100

print(f'\n{"="*55}')
print(f'  V4 SPECTRAL STEERING vs V3 vs TRANSFORMER')
print(f'{"="*55}')
for n, m in configs:
    p = count_params(m)
    eff = m.effective_params() if hasattr(m, 'effective_params') else p
    print(f'  {n:15}: {p:>8,} params (effective: {eff:,})')

print(f'\n  {"Config":15} {"Params":>8} {"Step":>5} {"TestLoss":>9} {"TestAcc":>8} {"TestPPL":>8}')
print(f'  {"-"*55}')
results = {}
for name, model in configs:
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_acc = 0
    for step in range(STEPS):
        model.train(); idx = torch.randint(0, td.shape[0], (B,)); batch = td[idx]
        loss = F.cross_entropy(model(batch[:, :4]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            tl = evaluate(model, vd); ac = accuracy(model, vd); best_acc = max(best_acc, ac)
            print(f'  {name:15} {count_params(model):>8,} {step:5d} {tl:9.4f} {ac:7.1f}% {math.exp(tl):8.2f}')
    tl = evaluate(model, vd); ac = accuracy(model, vd); best_acc = max(best_acc, ac)
    results[name] = {'loss': tl, 'acc': ac, 'best_acc': best_acc, 'ppl': math.exp(tl)}

print(f'\n{"="*55}')
print(f'  FINAL: V3 vs V4 vs Transformer')
print(f'{"="*55}')
print(f'  {"Config":15} {"Params":>8} {"TestAcc":>8} {"BestAcc":>8} {"TestPPL":>8}')
print(f'  {"-"*45}')
for name, model in configs:
    r = results[name]; print(f'  {name:15} {count_params(model):>8,} {r["acc"]:>7.1f}% {r["best_acc"]:>7.1f}% {r["ppl"]:>8.2f}')
