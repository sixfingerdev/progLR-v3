"""Hybrid scaling: V3 + 1/2 Standard layers at larger D=256, L=8 arithmetic test.
Hypothesis: deeper V3 provides richer representations → single std layer reasons better."""
import sys, os, math, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from condenser.hybrid_model import HybridTransformer
from condenser.baseline_transformer import BaselineTransformer
from condenser.genomatrix import _R_POOL, GenoTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

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

B = 64; STEPS = 5000

configs = [
    # D=192, L=6: moderate scale
    ('D192-L6 V3', GenoTransformer(VOCAB, 192, 6, 6, 8, 1.0)),
    ('D192-L6 H1', HybridTransformer(VOCAB, 192, 6, 6, 8, 1.0, 1)),
    ('D192-L6 H2', HybridTransformer(VOCAB, 192, 6, 6, 8, 1.0, 2)),
    ('D192-L6 TF', BaselineTransformer(VOCAB, 192, 6, 6)),
]

def count_params(m):
    return m.count_trained() if hasattr(m,'count_trained') else m.count_params()

def acc(model, data):
    model.eval(); c=0; t=0
    with torch.no_grad():
        for i in range(0,data.shape[0],B):
            b = data[i:i+B]; preds = model(b[:,:4]).argmax(dim=-1)
            c += (preds[:,-1]==b[:,-1]).sum().item(); t+=b.shape[0]
    return c/t*100

def eval_loss(model, data):
    model.eval(); t=0; n=0
    with torch.no_grad():
        for i in range(0,data.shape[0],B):
            b = data[i:i+B]
            t += F.cross_entropy(model(b[:,:4]).reshape(-1,VOCAB),b[:,1:].reshape(-1)).item()*b.shape[0]; n+=b.shape[0]
    return t/n

print(f'\n{"="*70}')
print(f'  HYBRID SCALING TEST: V3 + Standard at larger D')
print(f'{"="*70}')
results = {}
for name, model in configs:
    _R_POOL.clear()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_acc = 0
    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, td.shape[0], (B,)); batch = td[idx]
        loss = F.cross_entropy(model(batch[:,:4]).reshape(-1,VOCAB),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            ac = acc(model, vd); best_acc = max(best_acc, ac)
    ac = acc(model, vd); best_acc = max(best_acc, ac)
    tl = eval_loss(model, vd)
    p = count_params(model)
    results[name] = {'acc': ac, 'best_acc': best_acc, 'ppl': math.exp(tl), 'params': p}
    print(f'  {name:20} {p:>8,} {STEPS:5d} step | Acc: {ac:>5.1f}% (best {best_acc:>5.1f}%) | PPL: {math.exp(tl):.2f}')

# Now test deeper: L=12 with D=192
print(f'\n=== DEEPER TEST: D=192, L=12 ===')
deeper_configs = [
    ('D192-L12 V3', GenoTransformer(VOCAB, 192, 12, 6, 8, 1.0)),
    ('D192-L12 H1', HybridTransformer(VOCAB, 192, 12, 6, 8, 1.0, 1)),
    ('D192-L12 H2', HybridTransformer(VOCAB, 192, 12, 6, 8, 1.0, 2)),
    ('D192-L12 TF', BaselineTransformer(VOCAB, 192, 12, 6)),
]
for name, model in deeper_configs:
    _R_POOL.clear()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_acc = 0
    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, td.shape[0], (B,)); batch = td[idx]
        loss = F.cross_entropy(model(batch[:,:4]).reshape(-1,VOCAB),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            ac = acc(model, vd); best_acc = max(best_acc, ac)
    ac = acc(model, vd); best_acc = max(best_acc, ac)
    p = count_params(model)
    results[name] = {'acc': ac, 'best_acc': best_acc, 'params': p}
    print(f'  {name:20} {p:>8,} {STEPS:5d} step | Acc: {ac:>5.1f}% (best {best_acc:>5.1f}%)')

print(f'\n{"="*70}')
print(f'  COMPRESSION vs REASONING at scale')
print(f'{"="*70}')
print(f'  {"Config":20} {"Params":>8} {"Acc":>6} {"Comp vs TF":>11}')
print(f'  {"-"*50}')
for n, r in sorted(results.items()):
    tf_key = [k for k in results.keys() if k.endswith('TF') and n.split(' ')[0] in k]
    if n.endswith('TF'):
        print(f'  {n:20} {r["params"]:>8,} {r["acc"]:>5.1f}% {"1×":>11}')
    else:
        tf_p = results.get([k for k in results.keys() if k.endswith('TF') and n.split(' ')[0] in k][0], {}).get('params', 1) if any(k.endswith('TF') and n.split(' ')[0] in k for k in results.keys()) else 1
        # compute tf_p
        tf_key_matches = [k for k in results.keys() if k.endswith('TF') and n.split(' ')[0] in k]
        if tf_key_matches:
            comp = results[tf_key_matches[0]]['params'] / max(r['params'], 1)
            print(f'  {n:20} {r["params"]:>8,} {r["acc"]:>5.1f}% {comp:>10.1f}×')
