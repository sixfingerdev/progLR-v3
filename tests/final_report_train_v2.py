"""
D=768 Progressive Low-Rank on WikiText-103. FP32, B=16, stable.
v2: SAME math as final_report_train.py, but the 50K head is computed
chunked with ANALYTIC gradients (dH/dU/dV) inside no_grad + a single
h.backward(dH) through the layers. No 1.6GB logits tensor, no softmax
graph nodes, no cudaMalloc churn. Layer/embedding backward flows via
autograd exactly once.
"""
import os, sys, math, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'condenser', 'wt103_data.pkl')
VOCAB = os.path.join(BASE, 'condenser', 'wt103_vocab.pkl')
OUT_DIR = os.path.join(BASE, 'report_out_v2')
os.makedirs(OUT_DIR, exist_ok=True)

# === Low-Rank Layer ===
class LRLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, U, V, a, r):
        s=a/r; W=s*(U@V.t())
        ctx.save_for_backward(x,U,V); ctx.s=s
        return F.linear(x, W)
    @staticmethod
    def backward(ctx, go):
        x,U,V=ctx.saved_tensors
        i_f=x.shape[-1]; o_f=go.shape[-1]
        x_f=x.reshape(-1,i_f); go_f=go.reshape(-1,o_f)
        gW=go_f.t()@x_f; s=ctx.s
        return (F.linear(go,s*(V@U.t())), s*(gW@V), s*(gW.t()@U), None, None)

class LRLayer(nn.Module):
    def __init__(self, dim, heads, r=16, a=64):
        super().__init__()
        self.h,self.dh,self.r,self.a=heads,dim//heads,r,a
        for nm,od,id_ in [('q',dim,dim),('k',dim,dim),('v',dim,dim),('o',dim,dim),('ff1',4*dim,dim),('ff2',dim,4*dim)]:
            setattr(self,f'U_{nm}',nn.Parameter(torch.randn(od,r)*0.02))
            setattr(self,f'V_{nm}',nn.Parameter(torch.randn(id_,r)*0.02))
        self.n1,self.n2=nn.LayerNorm(dim),nn.LayerNorm(dim)
    def _l(self,x,nm):
        return LRLinearFn.apply(x,getattr(self,f'U_{nm}'),getattr(self,f'V_{nm}'),self.a,self.r)
    def forward(self,x):
        B,T,D=x.shape; h=self.dh; n=self.n1(x)
        Q=self._l(n,'q').view(B,T,self.h,h).transpose(1,2)
        K=self._l(n,'k').view(B,T,self.h,h).transpose(1,2)
        V_=self._l(n,'v').view(B,T,self.h,h).transpose(1,2)
        a=F.scaled_dot_product_attention(Q,K,V_,is_causal=True)
        o=a.transpose(1,2).reshape(B,T,D)
        x=x+self._l(o,'o'); n2=self.n2(x)
        x=x+self._l(F.relu(self._l(n2,'ff1')),'ff2')
        return x

from condenser.v3_embed import V3WeightEmbedding

class StdLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h=heads; self.dh=dim//heads
        self.W_Q=nn.Linear(dim,dim,bias=False); self.W_K=nn.Linear(dim,dim,bias=False)
        self.W_V=nn.Linear(dim,dim,bias=False); self.W_O=nn.Linear(dim,dim,bias=False)
        self.W1=nn.Linear(dim,dim*4,bias=False); self.W2=nn.Linear(dim*4,dim,bias=False)
        self.n1=nn.LayerNorm(dim); self.n2=nn.LayerNorm(dim)
    def forward(self,x):
        B,T,D=x.shape; h=self.h; dh=self.dh; n=self.n1(x)
        Q=self.W_Q(n).view(B,T,h,dh).transpose(1,2)
        K=self.W_K(n).view(B,T,h,dh).transpose(1,2)
        V=self.W_V(n).view(B,T,h,dh).transpose(1,2)
        a=F.scaled_dot_product_attention(Q,K,V,is_causal=True)
        x=x+self.W_O(a.transpose(1,2).reshape(B,T,D))
        n=self.n2(x)
        return x+self.W2(F.relu(self.W1(n)))

class ProgLR(nn.Module):
    def __init__(self, vs, D, L, H, num_std=1, r=16, a=64):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1, 1024, D) * 0.01)
        self.layers = nn.ModuleList()
        for i in range(L):
            if i < num_std:
                self.layers.append(StdLayer(D, H))
            else:
                self.layers.append(LRLayer(D, H, r, a))
        self.norm = nn.LayerNorm(D)
    def get_hidden(self, x):
        B,T=x.shape
        h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.norm(h)
    def forward(self, x):
        return self.emb.forward_head(self.get_hidden(x))

# === Chunked head with analytic gradients ===
CHUNK = 50000
def head_loss_fn(h, U, V, R, y, alpha, rank, lc_buf, sub_batch):
    """Analytic chunked head: loss value (detached) + exact dH, dU, dV.
    Math identical to original: l_ic = h_i.R_c + s*(h_i.V).U_c, L = sum CE.
    forward: lc = h_b @ Rc.t() + s*(t_b @ Uc.t())  -> (n, CHUNK)"""
    s = alpha / rank
    N, D = h.shape
    dH = torch.zeros_like(h); dU = torch.zeros_like(U); dV = torch.zeros_like(V)
    t = h @ V
    total = 0.0
    y = y.reshape(-1)
    nch = math.ceil(R.shape[0] / CHUNK)
    with torch.no_grad():
        for i0 in range(0, N, sub_batch):
            i1 = min(N, i0 + sub_batch)
            n = i1 - i0
            h_b = h[i0:i1]; t_b = t[i0:i1]; y_b = y[i0:i1]
            lc = lc_buf[:n]
            for c in range(nch):
                Rc = R[c*CHUNK:(c+1)*CHUNK]
                Uc = U[c*CHUNK:(c+1)*CHUNK]
                torch.addmm(lc, h_b, Rc.t(), beta=0, alpha=1, out=lc)
                torch.addmm(lc, t_b, Uc.t(), beta=1, alpha=s, out=lc)
                yc = torch.where((y_b >= c*CHUNK) & (y_b < (c+1)*CHUNK), y_b - c*CHUNK, -100)
                valid = yc >= 0
                nv = valid.sum().item()
                if nv == 0:
                    continue
                total += F.cross_entropy(lc, yc.long(), reduction='sum').item()
                g = F.softmax(lc, dim=-1)
                tg = yc.clamp_min(0).long()
                g.scatter_add_(1, tg.unsqueeze(1), torch.full_like(g, -1.0))
                g[~valid] = 0.0
                dU[c*CHUNK:(c+1)*CHUNK].add_(s * (g.t() @ t_b))
                dH[i0:i1].add_(g @ Rc)
                dH[i0:i1].add_((s * (g @ Uc)) @ V.t())
                dV.add_(s * (h_b.t() @ (g @ Uc)))
    return total, dH, dU, dV

# === Load data ===
print('Loading data...', flush=True)
with open(DATA, 'rb') as f:
    d = pickle.load(f)
VS = d['vocab_size']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tr_x = torch.tensor(d['tr_x'], device=dev, dtype=torch.int32)
tr_y = torch.tensor(d['tr_y'], device=dev, dtype=torch.int32)
te_x = torch.tensor(d['te_x'], device=dev, dtype=torch.int32)
te_y = torch.tensor(d['te_y'], device=dev, dtype=torch.int32)
del d

with open(VOCAB, 'rb') as f:
    voc = pickle.load(f)
id2word = voc['id2word']

# === Setup ===
B=16; D=768; H=12; R=16; A=64; SEQ=512
print(f'Vocab: {VS}, Train: {tr_x.shape[0]:,} seq, Valid: {te_x.shape[0]:,} seq', flush=True)

model = ProgLR(VS, D, 2, H, 1, R, A)
model.to(dev)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {param_count:,}', flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
S = int(os.environ.get('V2_STEPS', '20000'))
CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

# === Helpers ===
def eval_ppl(model, xs, ys):
    model.eval(); tot_s=0.0; tot_v=0
    lc = torch.empty(4096, CHUNK, device=dev)
    with torch.no_grad():
        for i in range(0, xs.shape[0], 8):
            bx=xs[i:i+8]; by=ys[i:i+8]
            h = model.get_hidden(bx).reshape(-1, D)
            s = model.emb.alpha / model.emb.rank
            t = h @ model.emb.V
            y = by.reshape(-1)
            n = h.shape[0]
            for c in range(math.ceil(VS / CHUNK)):
                Rc = model.emb.R[c*CHUNK:(c+1)*CHUNK]
                Uc = model.emb.U[c*CHUNK:(c+1)*CHUNK]
                torch.addmm(lc[:n], h, Rc.t(), beta=0, alpha=1, out=lc[:n])
                torch.addmm(lc[:n], t, Uc.t(), beta=1, alpha=s, out=lc[:n])
                yc = torch.where((y >= c*CHUNK) & (y < (c+1)*CHUNK), y - c*CHUNK, -100)
                valid = yc >= 0
                nv = valid.sum().item()
                if nv > 0:
                    tot_s += F.cross_entropy(lc[:n], yc.long(), reduction='sum').item()
                    tot_v += nv
    return math.exp(tot_s / max(tot_v, 1))

def generate(model, prompt_tokens, max_new=30):
    model.eval()
    inp = torch.tensor([prompt_tokens], device=dev, dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(inp[:, -SEQ:])
            nxt = logits[0, -1].argmax().item()
            inp = torch.cat([inp, torch.tensor([[nxt]], device=dev)], dim=1)
    return [id2word.get(t.item(), '<unk>') for t in inp[0]]

def main():
    v2i = {v: k for k, v in id2word.items()}
    prompt_ids = [v2i.get('the', 2)]
    print(f'\nStarting training: B={B}, D={D}, steps={S}\n', flush=True)
    t_start = time.time()
    best_ppl=float('inf')
    lc_buf = torch.empty(2048, CHUNK, device=dev)

    for st in range(S):
        idx = torch.randint(0, tr_x.shape[0], (B,), device=dev)
        bx, by = tr_x[idx], tr_y[idx]
        h = model.get_hidden(bx)
        tot_s, dH, dU, dV = head_loss_fn(h.reshape(-1, D), model.emb.U, model.emb.V,
                                         model.emb.R, by, model.emb.alpha, model.emb.rank,
                                         lc_buf, 2048)
        loss = tot_s / (B * SEQ)
        h.reshape(-1, D).backward(dH)
        with torch.no_grad():
            if model.emb.U.grad is None: model.emb.U.grad = dU
            else: model.emb.U.grad.add_(dU)
            if model.emb.V.grad is None: model.emb.V.grad = dV
            else: model.emb.V.grad.add_(dV)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if st % 10 == 0:
            elapsed = time.time()-t_start
            tok_per_sec = (st+1)*B*SEQ/elapsed if elapsed > 0 else 0
            print(f'  st={st:>5} | loss={loss:.3f} | tok/s={tok_per_sec:.0f} | elapsed={elapsed:.0f}s', flush=True)

        if st > 0 and st % 2000 == 0:
            ppl=eval_ppl(model, te_x, te_y); best_ppl=min(best_ppl,ppl)
            gen_words = generate(model, prompt_ids, 30)
            gen_text = ' '.join(gen_words).replace(' @', '').replace(' ,', ',').replace(' .', '.')
            print(f'  >>> EVAL st={st:>5} | val_ppl={ppl:.2f} | best={best_ppl:.2f}', flush=True)
            print(f'  >>> GEN: {gen_text}', flush=True)
            ckpt = {'step': st, 'model_state': model.state_dict(), 'optim_state': opt.state_dict(),
                    'best_ppl': best_ppl, 'loss': loss}
            torch.save(ckpt, os.path.join(CKPT_DIR, f'ckpt_{st:05d}.pt'))
            print(f'  >>> Saved checkpoint {st:05d}', flush=True)

    ppl=eval_ppl(model, te_x, te_y); best_ppl=min(best_ppl,ppl)
    gen_words = generate(model, prompt_ids, 30)
    gen_text = ' '.join(gen_words).replace(' @', '').replace(' ,', ',').replace(' .', '.')
    duration = time.time()-t_start
    print(f'Done! {duration:.0f}s | val_ppl={ppl:.2f} | best_ppl={best_ppl:.2f}', flush=True)

    report = f"""# V3-Embed + Progressive Low-Rank: Technical Report (v2 optimized)

## Model
- **D={D}, H={H}, V={VS:,}**: 1 Std TF + 1 LR (r={R})
- **Params**: {param_count:,}
- **Batch**: {B} × {SEQ}

## Dataset
- **WikiText-103**: {tr_x.shape[0]:,} train seq, {te_x.shape[0]:,} valid seq

## Training
- **AdamW** (lr=3e-4, wd=0.01), FP32, {S} steps
- **Duration**: {duration:.0f}s, **Throughput**: {S*B*SEQ/duration:.0f} tok/s

## Results
- **Best Validation PPL**: {best_ppl:.2f}
- **Final Validation PPL**: {ppl:.2f}

## Sample
{gen_text}

## Architecture
- **V3-Embed**: {VS*R + D*R:,} params
- **Standard TF**: {4*D*D + 2*D*4*D:,} params
- **Low-Rank (r={R})**: {4*(D*R+D*R) + 2*(D*R+4*D*R):,} params
"""
    with open(os.path.join(OUT_DIR,'report.md'),'w') as f: f.write(report)
    print('Saved.', flush=True)

if __name__ == '__main__':
    main()
