"""V5 — Nonlinear Side Paths: arithmetic reasoning + WikiText."""
import sys, os, math, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from condenser.genomatrix_v5 import GenoTransformerV5
from condenser.genomatrix import GenoTransformer as V3
from condenser.baseline_transformer import BaselineTransformer
from condenser.genomatrix_v5 import _R_POOL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# === ARITHMETIC ===
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
td_arith = make_data(indices[:split]); vd_arith = make_data(indices[split:])
print(f'Arithmetic: train {td_arith.shape[0]}, test {vd_arith.shape[0]}')

# === WIKITEXT ===
import re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def read_text(p):
    with open(p, 'r', encoding='utf-8') as f: return f.read()
def tok(t):
    return re.sub(r'[^a-z0-9\s\'\.,!?;:]', ' ', t.lower()).split()
tw = tok(read_text(r'C:\Users\enesa\train.txt'))
vw = tok(read_text(r'C:\Users\enesa\valid.txt'))
from collections import Counter
vl = ['<pad>','<unk>','<bos>','<eos>'] + [w for w,_ in Counter(tw).most_common(5000)]
WIKI_VOCAB = len(vl); w2i = {w:i for i,w in enumerate(vl)}
SEQ = 64
def enc(words):
    ids = [w2i['<bos>']] + [w2i.get(w,w2i['<unk>']) for w in words] + [w2i['<eos>']]
    d = torch.tensor(ids)
    ex = []
    for i in range(0,len(d)-SEQ-1,SEQ//2):
        ex.append(d[i:i+SEQ+1])
    return torch.stack(ex).to(device) if ex else torch.zeros((0,SEQ+1),dtype=torch.long).to(device)
td_wiki = enc(tw); vd_wiki = enc(vw)
print(f'WikiText: train {td_wiki.shape[0]}, val {vd_wiki.shape[0]}, vocab {WIKI_VOCAB}')

D, L, HEADS, B = 128, 4, 4, 64
ARITH_STEPS = 5000
WIKI_STEPS = 3000

configs = [
    ('V3(r=8)', V3(102, D, L, HEADS, 8, 1.0)),
    ('V5(r=8,s=8)', GenoTransformerV5(102, D, L, HEADS, 8, 1.0, 8)),
    ('V5(r=8,s=16)', GenoTransformerV5(102, D, L, HEADS, 8, 1.0, 16)),
    ('V5(r=8,s=32)', GenoTransformerV5(102, D, L, HEADS, 8, 1.0, 32)),
    ('Transformer', BaselineTransformer(102, D, L, HEADS)),
]

def count_params(m):
    return m.count_trained() if hasattr(m,'count_trained') else m.count_params()

def eval_arith(model, data):
    model.eval(); t=0; n=0
    with torch.no_grad():
        for i in range(0,data.shape[0],B):
            b = data[i:i+B]
            t += F.cross_entropy(model(b[:,:4]).reshape(-1,102),b[:,1:].reshape(-1)).item()*b.shape[0]; n+=b.shape[0]
    return t/n

def acc_arith(model, data):
    model.eval(); c=0; t=0
    with torch.no_grad():
        for i in range(0,data.shape[0],B):
            b = data[i:i+B]; preds = model(b[:,:4]).argmax(dim=-1)
            c += (preds[:,-1]==b[:,-1]).sum().item(); t+=b.shape[0]
    return c/t*100

def eval_wiki(model, data):
    model.eval(); t=0; n=0
    with torch.no_grad():
        for i in range(0,data.shape[0],B):
            b = data[i:i+B]
            t += F.cross_entropy(model(b[:,:-1]).reshape(-1,WIKI_VOCAB),b[:,1:].reshape(-1)).item()*b.shape[0]; n+=b.shape[0]
    return t/n

print(f'\n{"="*60}')
print(f'  V5 TEST: Arithmetic Reasoning')
print(f'{"="*60}')
for n,m in configs:
    print(f'  {n:18}: {count_params(m):>8,} params')
print(f'\n  {"Config":18} {"Step":>5} {"TestAcc":>8} {"TestPPL":>8}')
print(f'  {"-"*45}')
arith_results = {}
for name, model in configs:
    _R_POOL.clear()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_acc = 0
    for step in range(ARITH_STEPS):
        model.train()
        idx = torch.randint(0, td_arith.shape[0], (B,)); batch = td_arith[idx]
        loss = F.cross_entropy(model(batch[:,:4]).reshape(-1,102),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step%500==0:
            ac = acc_arith(model, vd_arith); best_acc = max(best_acc, ac)
    ac = acc_arith(model, vd_arith); best_acc = max(best_acc, ac)
    tl = eval_arith(model, vd_arith)
    p = count_params(model)
    arith_results[name] = {'acc': ac, 'best_acc': best_acc, 'ppl': math.exp(tl)}
    print(f'  {name:18} {ARITH_STEPS:5d} {ac:>7.1f}% {math.exp(tl):>8.2f}')

# Test best V5 on WikiText
print(f'\n{"="*60}')
print(f'  V5 on WIKITEXT (language quality check)')
print(f'{"="*60}')
wiki_configs = [
    ('V3(r=8)', V3(WIKI_VOCAB, 384, 6, 6, 8, 1.0)),
    ('V5(r=8,s=16)', GenoTransformerV5(WIKI_VOCAB, 384, 6, 6, 8, 1.0, 16)),
    ('Transformer', BaselineTransformer(WIKI_VOCAB, 384, 6, 6)),
]
for n,m in wiki_configs:
    print(f'  {n:18}: {count_params(m):>8,} params')
print(f'\n  {"Config":18} {"Step":>5} {"ValPPL":>8}')
print(f'  {"-"*35}')
wiki_results = {}
for name, model in wiki_configs:
    _R_POOL.clear()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for step in range(WIKI_STEPS):
        model.train()
        idx = torch.randint(0, td_wiki.shape[0], (32,)); batch = td_wiki[idx]
        loss = F.cross_entropy(model(batch[:,:-1]).reshape(-1,WIKI_VOCAB),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step%500==0:
            vl = eval_wiki(model, vd_wiki)
    vl = eval_wiki(model, vd_wiki)
    p = count_params(model)
    tfp = count_params(BaselineTransformer(WIKI_VOCAB,384,6,6))
    comp = tfp/max(p,1)
    wiki_results[name] = {'val_ppl': math.exp(vl), 'comp': comp}
    print(f'  {name:18} {WIKI_STEPS:5d} {math.exp(vl):>8.2f} ({comp:.1f}x)')

print(f'\n{"="*60}')
print(f'  V5 COMPLETE RESULTS')
print(f'{"="*60}')
print(f'  {"Config":18} {"Params":>8} {"ArithAcc":>9} {"WikiPPL":>8} {"Comp":>6}')
print(f'  {"-"*55}')
for n,m in configs:
    ar = arith_results.get(n, {})
    wr = wiki_results.get(n, {})
    p = count_params(m)
    aa = ar.get('acc', 0)
    wp = wr.get('val_ppl', 0)
    comp = wr.get('comp', 0)
    print(f'  {n:18} {p:>8,} {aa:>8.1f}% {wp:>8.2f} {comp:>5.1f}x' if wp > 0 else f'  {n:18} {p:>8,} {aa:>8.1f}% {"-":>8} {"-":>6}')
