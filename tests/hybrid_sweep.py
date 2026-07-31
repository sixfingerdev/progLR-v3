"""Hybrid ratio sweep: optimal V3:Standard mix for reasoning vs compression."""
import sys, os, math, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from condenser.hybrid_model import HybridTransformer
from condenser.baseline_transformer import BaselineTransformer

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
print(f'Arithmetic: train {td.shape[0]}, test {vd.shape[0]}, vocab {VOCAB}')

D, L, HEADS, B, STEPS = 128, 6, 4, 64, 3000

def evaluate(model, data):
    model.eval(); t = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]
            t += F.cross_entropy(model(b[:,:4]).reshape(-1,VOCAB), b[:,1:].reshape(-1)).item()*b.shape[0]; n+=b.shape[0]
    return t/n

def accuracy(model, data):
    model.eval(); c = 0; t = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]; preds = model(b[:,:4]).argmax(dim=-1)
            c += (preds[:,-1] == b[:,-1]).sum().item(); t += b.shape[0]
    return c/t*100

from condenser.genomatrix import _R_POOL

# === TEST 1: Standard layer count sweep (r=8) ===
print(f'\n=== SWEEP: Standard layers in L={L} (r=8) ===')
print(f'{"Config":20} {"Params":>8} {"Std%":>6} {"TestAcc":>8} {"TestPPL":>8} {"Comp":>6}')
print(f'{"-"*60}')
sw_results = []
for nstd in range(0, L+1):
    name = f'{L-nstd}V3+{nstd}Std'
    if nstd == L:
        model = BaselineTransformer(VOCAB, D, L, HEADS)
    else:
        model = HybridTransformer(VOCAB, D, L, HEADS, rank=8, alpha=1.0, num_standard=nstd)

    _R_POOL.clear()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for step in range(STEPS):
        model.train();
        idx = torch.randint(0, td.shape[0], (B,)); batch = td[idx]
        loss = F.cross_entropy(model(batch[:,:4]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    ac = accuracy(model, vd)
    tl = evaluate(model, vd)
    p = model.count_trained() if hasattr(model, 'count_trained') else sum(p.numel() for p in model.parameters() if p.requires_grad)
    tf_params = sum(p.numel() for p in BaselineTransformer(VOCAB, D, L, HEADS).parameters() if p.requires_grad)
    comp = tf_params / max(p, 1)
    sw_results.append({'name': name, 'nstd': nstd, 'params': p, 'acc': ac, 'ppl': math.exp(tl), 'comp': comp})
    std_pct = nstd / L * 100
    print(f'  {name:20} {p:>8,} {std_pct:5.0f}% {ac:>7.1f}% {math.exp(tl):>8.2f} {comp:>5.1f}x')

# === TEST 2: Rank sweep for best configs ===
print(f'\n=== RANK SWEEP (V3+1Std) ===')
for rank in [4, 8, 16, 32]:
    model = HybridTransformer(VOCAB, D, L, HEADS, rank=rank, alpha=1.0, num_standard=1)
    _R_POOL.clear()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for step in range(STEPS):
        model.train();
        idx = torch.randint(0, td.shape[0], (B,)); batch = td[idx]
        loss = F.cross_entropy(model(batch[:,:4]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    ac = accuracy(model, vd); tl = evaluate(model, vd)
    p = model.count_trained() if hasattr(model, 'count_trained') else sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  V3+1Std(r={rank:2d})  {p:>8,} {ac:>7.1f}% {math.exp(tl):>8.2f}')

print(f'\n=== RANK SWEEP (V3+2Std) ===')
for rank in [4, 8, 16, 32]:
    model = HybridTransformer(VOCAB, D, L, HEADS, rank=rank, alpha=1.0, num_standard=2)
    _R_POOL.clear()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for step in range(STEPS):
        model.train();
        idx = torch.randint(0, td.shape[0], (B,)); batch = td[idx]
        loss = F.cross_entropy(model(batch[:,:4]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    ac = accuracy(model, vd); tl = evaluate(model, vd)
    p = model.count_trained() if hasattr(model, 'count_trained') else sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  V3+2Std(r={rank:2d})  {p:>8,} {ac:>7.1f}% {math.exp(tl):>8.2f}')

# === SUMMARY ===
print(f'\n{"="*60}')
print(f'  HYBRID TRADE-OFF CURVE (r=8, L={L})')
print(f'{"="*60}')
print(f'  {"Config":20} {"Std%":>6} {"Compression":>12} {"TestAcc":>8}')
print(f'  {"-"*50}')
for r in sw_results:
    print(f'  {r["name"]:20} {r["nstd"]/L*100:5.0f}% {r["comp"]:>11.1f}x {r["acc"]:>7.1f}%')
