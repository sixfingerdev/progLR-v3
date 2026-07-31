"""
WikiText-2 benchmark + D=384 scaling for Progressive Low-Rank.
Configs:
  1. Baseline V3-Embed+TF L=2 (D=256)
  2. Prog 1Std+1LR r=16 L=2 (D=256)
  3. Prog 2Std+2LR r=16 L=4 (D=256)
  4. Prog 1Std+1LR r=16 L=2 (D=384) — on arithmetic only
  5. Prog 2Std+2LR r=16 L=4 (D=384) — on arithmetic only
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from condenser.v3_embed import V3WeightEmbedding
from condenser.baseline_transformer import BaselineTransformerLayer

# === Low-Rank ===
class LRLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, U, V, a, r):
        s=a/r; ctx.save_for_backward(x,U,V); ctx.s=s
        return F.linear(x, s*(U@V.t()))
    @staticmethod
    def backward(ctx, go):
        x,U,V=ctx.saved_tensors; i_f=x.shape[-1]; o_f=go.shape[-1]
        x_f=x.reshape(-1,i_f); go_f=go.reshape(-1,o_f); gW=go_f.t()@x_f; s=ctx.s
        return (F.linear(go,s*(V@U.t())), s*(gW@V), s*(gW.t()@U), None, None)

class LRLayer(nn.Module):
    def __init__(self, dim, heads, r=16, a=64, causal=True):
        super().__init__()
        self.h,self.dh,self.r,self.a,self.causal=heads,dim//heads,r,a,causal
        for nm,od,id_ in [('q',dim,dim),('k',dim,dim),('v',dim,dim),('o',dim,dim),('ff1',4*dim,dim),('ff2',dim,4*dim)]:
            setattr(self,f'U_{nm}',nn.Parameter(torch.randn(od,r)*0.02))
            setattr(self,f'V_{nm}',nn.Parameter(torch.randn(id_,r)*0.02))
        self.n1,self.n2=nn.LayerNorm(dim),nn.LayerNorm(dim)
    def _l(self,x,nm): return LRLinearFn.apply(x,getattr(self,f'U_{nm}'),getattr(self,f'V_{nm}'),self.a,self.r)
    def forward(self,x):
        B,T,D=x.shape; h=self.dh; n=self.n1(x)
        Q=self._l(n,'q').view(B,T,self.h,h).transpose(1,2)
        K=self._l(n,'k').view(B,T,self.h,h).transpose(1,2)
        V=self._l(n,'v').view(B,T,self.h,h).transpose(1,2)
        a=Q@K.transpose(-2,-1)/math.sqrt(h)
        if self.causal:
            mask=torch.triu(torch.full((T,T),float('-inf'),device=x.device),diagonal=1)
            a=a+mask
        a=F.softmax(a,dim=-1)
        o=(a@V).transpose(1,2).reshape(B,T,D)
        x=x+self._l(o,'o'); n2=self.n2(x)
        x=x+self._l(F.relu(self._l(n2,'ff1')),'ff2')
        return x

class Baseline(nn.Module):
    def __init__(self,vs,D,L,H,r=16,a=64):
        super().__init__()
        self.emb=V3WeightEmbedding(vs,D,r,a); self.pos=nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers=nn.ModuleList([BaselineTransformerLayer(D,H) for _ in range(L)])
        self.norm=nn.LayerNorm(D)
    def forward(self,x):
        B,T=x.shape; h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.emb.forward_head(self.norm(h))
    def cp(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ProgLR(nn.Module):
    def __init__(self,vs,D,L,H,num_std=1,r=16,a=64):
        super().__init__()
        self.emb=V3WeightEmbedding(vs,D,r,a); self.pos=nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers=nn.ModuleList()
        for i in range(L):
            if i<num_std: self.layers.append(BaselineTransformerLayer(D,H))
            else: self.layers.append(LRLayer(D,H,r,a))
        self.norm=nn.LayerNorm(D)
    def forward(self,x):
        B,T=x.shape; h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.emb.forward_head(self.norm(h))
    def cp(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===== WIKITEXT-2 =====
def load_wikitext2(vocab_size=5000, seq_len=64, dev='cuda'):
    from collections import Counter
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'condenser')
    # Read train and validation separately
    splits = {}
    for split_name in ['train', 'valid']:
        fp = os.path.join(path, f'{split_name}.txt')
        if not os.path.exists(fp):
            print(f'  WikiText-2 not found at {fp}', flush=True)
            return None, None, None
        with open(fp, 'r', encoding='utf-8') as f:
            text = f.read()
        text = re.sub(r'\s+', ' ', text).strip().lower()
        splits[split_name] = text.split()
    # Build vocab from full training set
    counts = Counter(splits['train'])
    most_common = counts.most_common(vocab_size - 2)
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, (w, _) in enumerate(most_common):
        vocab[w] = i + 2
    vs = vocab_size

    def tokenize_and_chunk(words, seq_len, max_words=None):
        if max_words: words = words[:max_words]
        ids = [vocab.get(w, 1) for w in words]
        xs = [ids[i:i+seq_len] for i in range(0, len(ids)-seq_len, seq_len)]
        ys = [ids[i+1:i+seq_len+1] for i in range(0, len(ids)-seq_len, seq_len)]
        return torch.tensor(xs, dtype=torch.long, device=dev), torch.tensor(ys, dtype=torch.long, device=dev)

    tr_x, tr_y = tokenize_and_chunk(splits['train'], seq_len)
    te_x, te_y = tokenize_and_chunk(splits['valid'], seq_len, 30000)
    return (tr_x, tr_y), (te_x, te_y), vs


def ppl_loss(model, x, y, vs):
    x = x.clamp(0, vs-1); y = y.clamp(0, vs-1)
    logits = model(x).reshape(-1, vs)
    return F.cross_entropy(logits, y.reshape(-1))

def eval_ppl(model, x, y, vs, B=64):
    model.eval(); losses=[]
    with torch.no_grad():
        for i in range(0, x.shape[0], B):
            bx, by = x[i:i+B], y[i:i+B]
            losses.append(ppl_loss(model, bx, by, vs).item())
    return math.exp(sum(losses)/len(losses))

def train_wikitext(model, train, test, steps=4000, B=64):
    tr_x, tr_y = train; te_x, te_y = test
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    vs = model.emb.R.shape[0]
    bo = float('inf')
    for st in range(steps):
        model.train()
        idx = torch.randint(0, tr_x.shape[0], (B,))
        loss = ppl_loss(model, tr_x[idx], tr_y[idx], vs)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if st % 1000 == 0:
            p = eval_ppl(model, te_x[:256], te_y[:256], vs)
            bo = min(bo, p)
            print(f'  step={st:>4} | loss={loss.item():.3f} | val_ppl={p:.1f}', flush=True)
    p = eval_ppl(model, te_x, te_y, vs, B); bo = min(bo, p)
    print(f'  FINAL | val_ppl={p:.1f} | best={bo:.1f}', flush=True)
    return bo


# ===== ARITHMETIC (for D=384 scaling) =====
def arith_data(N=100, dev='cuda'):
    V=N+2; P=N; E=N+1
    tr=torch.tensor([[a,P,b,E,(a+b)%N] for a in range(50) for b in range(N)], dtype=torch.long, device=dev)
    te=torch.tensor([[a,P,b,E,(a+b)%N] for a in range(50,N) for b in range(N)], dtype=torch.long, device=dev)
    return V, tr, te

def acc(model, data, B=256):
    model.eval(); c=0; t=0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b=data[i:i+B]; p=model(b[:,:4]).argmax(dim=-1)
            c+=(p[:,-1]==b[:,-1]).sum().item(); t+=b.shape[0]
    return c/t*100

def train_arith(model, tr, te, V, steps=10000, B=64):
    opt=torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    bo=0
    for st in range(steps):
        model.train()
        idx=torch.randint(0, tr.shape[0], (B,)); batch=tr[idx]
        loss=F.cross_entropy(model(batch[:,:4]).reshape(-1,V), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if st%2500==0:
            o=acc(model,te); bo=max(bo,o)
            print(f'  step={st:>4} | train={acc(model,tr):.1f}% | ood={o:.1f}%', flush=True)
    o=acc(model,te); bo=max(bo,o)
    print(f'  FINAL | ood={o:.1f}% | best={bo:.1f}%', flush=True)
    return bo


if __name__ == "__main__":
    from condenser.genomatrix import _R_POOL
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}', flush=True)

    # ===== 1. WikiText-2 Benchmark (D=256) =====
    print(f'\n{"="*60}', flush=True)
    print(f'  WIKITEXT-2 BENCHMARK (V=5000, D=256)', flush=True)
    print(f'{"="*60}', flush=True)
    train_data, test_data, vs = load_wikitext2(5000, 64, dev)
    if train_data is not None:
        tr_x, tr_y = train_data; te_x, te_y = test_data
        print(f'Vocab: {vs}, Train: {tr_x.shape}, Test: {te_x.shape}', flush=True)

        wt2_cfgs = [
            ('Baseline V3-Embed+TF L=2', Baseline(vs, 256, 2, 4, 16, 64)),
            ('Prog 1Std+1LR r=16 L=2', ProgLR(vs, 256, 2, 4, 1, 16, 64)),
        ]
        print(f'\n{"Model":<30} {"Params":>10}', flush=True)
        for n,m in wt2_cfgs: print(f'  {n:<28} {m.cp():>10,}', flush=True)

        wt2_results = {}
        for name, model in wt2_cfgs:
            _R_POOL.clear(); torch.cuda.reset_peak_memory_stats(dev)
            model.to(dev)
            print(f'\n  --- {name} ---', flush=True)
            best_ppl = train_wikitext(model, train_data, test_data, 4000, 64)
            wt2_results[name] = {'ppl': best_ppl, 'params': model.cp()}

        print(f'\n{"="*60}', flush=True)
        print(f'  WIKITEXT-2 RESULTS', flush=True)
        print(f'{"="*60}', flush=True)
        print(f'  {"Model":<30} {"Params":>9} {"Val PPL":>9}', flush=True)
        print(f'  {"-"*50}', flush=True)
        for n, r in sorted(wt2_results.items()):
            print(f'  {n:<30} {r["params"]:>9,} {r["ppl"]:>8.1f}', flush=True)
    else:
        print(f'  WikiText-2 not found, skipping.', flush=True)

    # ===== SUMMARY =====
    print(f'\n{"="*70}', flush=True)
    print(f'  WIKITEXT-2 RESULTS', flush=True)
    print(f'{"="*70}', flush=True)
    if train_data is not None:
        for n, r in sorted(wt2_results.items(), key=lambda x: x[1]['ppl']):
            print(f'  {n:<30} | {r["params"]:>8,} params | {r["ppl"]:>5.1f} PPL', flush=True)
