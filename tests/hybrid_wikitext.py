"""WikiText-2 word-level: Hybrid vs V3 vs Transformer on LANGUAGE."""
import sys, os, math, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from condenser.hybrid_model import HybridTransformer
from condenser.genomatrix import GenoTransformer
from condenser.baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_text(p):
    with open(p, 'r', encoding='utf-8') as f: return f.read()
def tok(t):
    return re.sub(r'[^a-z0-9\s\'\.,!?;:]', ' ', t.lower()).split()

tw = tok(read_text(r'C:\Users\enesa\train.txt'))
vw = tok(read_text(r'C:\Users\enesa\valid.txt'))
tw2 = tok(read_text(r'C:\Users\enesa\law\tests\wiki.test.tokens'))
print(f'Tokens: train {len(tw):,}, val {len(vw):,}, test {len(tw2):,}')

from collections import Counter
vl = ['<pad>','<unk>','<bos>','<eos>'] + [w for w,_ in Counter(tw).most_common(5000)]
VOCAB = len(vl); w2i = {w:i for i,w in enumerate(vl)}

SEQ = 64
def enc(words):
    ids = [w2i['<bos>']] + [w2i.get(w,w2i['<unk>']) for w in words] + [w2i['<eos>']]
    d = torch.tensor(ids)
    ex = []
    for i in range(0,len(d)-SEQ-1,SEQ//2):
        ex.append(d[i:i+SEQ+1])
    return torch.stack(ex).to(device) if ex else torch.zeros((0,SEQ+1),dtype=torch.long).to(device)
td = enc(tw); vd = enc(vw); ted = enc(tw2)
print(f'Seqs: train {td.shape[0]}, val {vd.shape[0]}, test {ted.shape[0]}, vocab {VOCAB}')

D, L, HEADS, RANK, B, STEPS = 384, 6, 6, 8, 32, 3000

configs = [
    ('V3(6V3+0Std)', GenoTransformer(VOCAB, D, L, HEADS, RANK, alpha=1.0)),
    ('Hybrid(5V3+1Std)', HybridTransformer(VOCAB, D, L, HEADS, RANK, alpha=1.0, num_standard=1)),
    ('Hybrid(4V3+2Std)', HybridTransformer(VOCAB, D, L, HEADS, RANK, alpha=1.0, num_standard=2)),
    ('Transformer', BaselineTransformer(VOCAB, D, L, HEADS)),
]

def count_params(m):
    return m.count_trained() if hasattr(m,'count_trained') else m.count_params()

def evaluate(model, data):
    model.eval(); t=0; n=0
    with torch.no_grad():
        for i in range(0,data.shape[0],B):
            b = data[i:i+B]
            t += F.cross_entropy(model(b[:,:-1]).reshape(-1,VOCAB),b[:,1:].reshape(-1)).item()*b.shape[0]; n+=b.shape[0]
    return t/n

print(f'\n{"="*70}')
print(f'  WIKITEXT-2: Hybrid Language Modeling (D={D}, L={L})')
print(f'{"="*70}')
for n,m in configs:
    print(f'  {n:18}: {count_params(m):>8,} params')
    if 'V3' in n:
        print(f'  {"":18}  effective: {m.effective_params():,}')
print(f'  {"":18}  TF equivalent: {count_params(BaselineTransformer(VOCAB,D,L,HEADS)):,}')

print(f'\n  {"Config":18} {"Step":>5} {"ValPPL":>8} {"TestPPL":>8} {"VRAM":>7} {"Comp":>7}')
print(f'  {"-"*60}')
results = {}
from condenser.genomatrix import _R_POOL

for name, model in configs:
    _R_POOL.clear()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_val = float('inf')
    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, td.shape[0], (B,)); batch = td[idx]
        loss = F.cross_entropy(model(batch[:,:-1]).reshape(-1,VOCAB),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step%500==0:
            vl = evaluate(model, vd); best_val = min(best_val, vl)
    vl = evaluate(model, vd); best_val = min(best_val, vl)
    tel = evaluate(model, ted)
    vram = torch.cuda.max_memory_allocated(device)/1e6
    p = count_params(model)
    tfp = count_params(BaselineTransformer(VOCAB,D,L,HEADS))
    comp = tfp / max(p,1)
    results[name] = {'val_ppl': math.exp(vl), 'test_ppl': math.exp(tel), 'vram': vram, 'comp': comp}
    print(f'  {name:18} {"3000":>5} {math.exp(vl):>8.2f} {math.exp(tel):>8.2f} {vram:>6.0f}MB {comp:>6.1f}x')

print(f'\n{"="*70}')
print(f'  WIKITEXT FINAL: Language + Reasoning Trade-off')
print(f'{"="*70}')
print(f'  {"Config":18} {"Params":>8} {"Comp":>6} {"ValPPL":>8} {"TestPPL":>8} {"ReasonAcc":>8}')
print(f'  {"-"*60}')
# Add reasoning acc from arithmetic test (from previous run, approximate)
reason = {'V3(6V3+0Std)': 0.4, 'Hybrid(5V3+1Std)': 6.4, 'Hybrid(4V3+2Std)': 17.5, 'Transformer': 97.3}
for name, model in configs:
    r = results[name]
    reas = reason.get(name, 0)
    print(f'  {name:18} {count_params(model):>8,} {r["comp"]:>5.1f}x {r["val_ppl"]:>8.2f} {r["test_ppl"]:>8.2f} {reas:>7.1f}%')
