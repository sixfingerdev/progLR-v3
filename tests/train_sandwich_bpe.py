"""
WT103 BPE pretrain: Sandwich 4Std+8LR r=16, D=1024, L=12, V=50K (BPE).
Winner of the D=1024 OOD comparison (bench_sandwich_ood.py, sandwich r=16 for LM quality).

Uses the proven scale_compare_train.py stack: factorized LR layers (y = s*(x@V)@U.t()),
analytic chunked head (no 50K logits ever materialized), all layers checkpointed,
fp32+TF32, data on CPU. Same hyperparams as the scale comparison (lr 3e-4, wd 0.01,
clip 1.0, B=16, seq 512) for comparability.
"""
import os, sys, math, pickle, time, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.manual_seed(7)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'condenser', 'wt103_data_bpe.pkl')
OUT = os.path.join(BASE, 'report_out_sandwich_bpe')
os.makedirs(OUT, exist_ok=True)

D, H, R, A, SEQ = 1024, 16, 16, 64, 512
PATTERN = ['std', 'std'] + ['lr'] * 4 + ['std'] + ['lr'] * 4 + ['std']
B = int(os.environ.get('SW_B', '16'))
STEPS = int(os.environ.get('SW_STEPS', '10000'))
print(f'D={D} L={len(PATTERN)} (4Std+8LR r={R}) B={B} steps={STEPS}', flush=True)

from condenser.v3_embed import V3WeightEmbedding

class LRLayer(nn.Module):
    def __init__(self, dim, heads, r=16, a=64):
        super().__init__()
        self.h, self.dh, self.r, self.a = heads, dim // heads, r, a
        for nm, od, id_ in [('q', dim, dim), ('k', dim, dim), ('v', dim, dim), ('o', dim, dim), ('ff1', 4 * dim, dim), ('ff2', dim, 4 * dim)]:
            setattr(self, f'U_{nm}', nn.Parameter(torch.randn(od, r) * 0.02))
            setattr(self, f'V_{nm}', nn.Parameter(torch.randn(id_, r) * 0.02))
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    def _l(self, x, nm):
        s = self.a / self.r
        return s * ((x @ getattr(self, f'V_{nm}')) @ getattr(self, f'U_{nm}').t())
    def forward(self, x):
        B, T, D = x.shape; h = self.dh; n = self.n1(x)
        Q = self._l(n, 'q').view(B, T, self.h, h).transpose(1, 2)
        K = self._l(n, 'k').view(B, T, self.h, h).transpose(1, 2)
        V_ = self._l(n, 'v').view(B, T, self.h, h).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V_, is_causal=True)
        x = x + self._l(a.transpose(1, 2).reshape(B, T, D), 'o')
        n2 = self.n2(x)
        return x + self._l(F.relu(self._l(n2, 'ff1')), 'ff2')

class StdLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.dh = dim // heads
        self.W_Q = nn.Linear(dim, dim, bias=False); self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False); self.W_O = nn.Linear(dim, dim, bias=False)
        self.W1 = nn.Linear(dim, dim * 4, bias=False); self.W2 = nn.Linear(dim * 4, dim, bias=False)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
    def forward(self, x):
        B, T, D = x.shape; h = self.h; dh = self.dh; n = self.n1(x)
        Q = self.W_Q(n).view(B, T, h, dh).transpose(1, 2)
        K = self.W_K(n).view(B, T, h, dh).transpose(1, 2)
        V = self.W_V(n).view(B, T, h, dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        x = x + self.W_O(a.transpose(1, 2).reshape(B, T, D))
        n = self.n2(x)
        return x + self.W2(F.relu(self.W1(n)))

class Model(nn.Module):
    def __init__(self, vs, D, H, pattern, r=16, a=64):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1, 1024, D) * 0.01)
        self.layers = nn.ModuleList(
            StdLayer(D, H) if k == 'std' else LRLayer(D, H, r, a) for k in pattern)
        self.norm = nn.LayerNorm(D)
    def get_hidden(self, x):
        B, T = x.shape
        h = self.emb.forward_embed(x) + self.pos[:, :T]
        for l in self.layers:
            h = ckpt.checkpoint(l, h, use_reentrant=False)
        return self.norm(h)

CHUNK = 50000
def head_loss_fn(h, U, V, R, y, alpha, rank, lc_buf, sub_batch):
    s = alpha / rank
    N, D = h.shape
    dH = torch.zeros_like(h); dU = torch.zeros_like(U); dV = torch.zeros_like(V)
    hf = h; tf = hf @ V
    total = 0.0
    y = y.reshape(-1)
    nch = math.ceil(R.shape[0] / CHUNK)
    with torch.no_grad():
        for i0 in range(0, N, sub_batch):
            i1 = min(N, i0 + sub_batch)
            n = i1 - i0
            h_b = hf[i0:i1]; t_b = tf[i0:i1]; y_b = y[i0:i1]
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
                total += F.cross_entropy(lc.float(), yc.long(), reduction='sum').item()
                torch.softmax(lc, dim=-1, out=lc)
                g = lc
                tg = yc.clamp_min(0).long()
                g.scatter_add_(1, tg.unsqueeze(1), torch.full_like(g, -1.0))
                g[~valid] = 0.0
                dU[c*CHUNK:(c+1)*CHUNK].add_((s * (g.t() @ t_b)).float())
                dH[i0:i1].add_((g @ Rc).float())
                dH[i0:i1].add_(((s * (g @ Uc)).float() @ V.t()))
    return total, dH, dU, dV

print('Loading data...', flush=True)
import numpy as np
d = np.load(os.path.join(BASE, 'condenser', 'wt103_bpe.npz'))
VS = 50000
dev = torch.device('cuda')
tr_x = torch.from_numpy(d['tr_x'])
tr_y = torch.from_numpy(d['tr_y'])
te_x = torch.from_numpy(d['te_x'])
te_y = torch.from_numpy(d['te_y'])

model = Model(VS, D, H, PATTERN, R, A).to(dev)
pc = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {pc:,} (V={VS})', flush=True)

RESUME = os.environ.get('SW_RESUME', '')
START = 0
best = float('inf')
if RESUME:
    ck = torch.load(os.path.join(OUT, RESUME), map_location=dev)
    model.load_state_dict(ck['model_state'])
    best = ck['best_ppl']; START = ck['step'] + 1
    print(f'>>> Resumed from {RESUME} | step={ck["step"]} | best_ppl={best:.2f}', flush=True)

def eval_ppl():
    model.eval()
    lc = torch.empty(1024, CHUNK, device=dev)
    tot_s = 0.0; tot_v = 0
    with torch.no_grad():
        for i in range(0, te_x.shape[0], 8):
            bx = te_x[i:i+8].to(dev); by = te_y[i:i+8].to(dev)
            h = model.get_hidden(bx).reshape(-1, D)
            t = h @ model.emb.V
            y = by.reshape(-1)
            n = h.shape[0]
            for i0 in range(0, n, 1024):
                i1 = min(n, i0 + 1024); nb = i1 - i0
                lc_n = lc[:nb]
                for c in range(math.ceil(VS / CHUNK)):
                    Rc = model.emb.R[c*CHUNK:(c+1)*CHUNK]; Uc = model.emb.U[c*CHUNK:(c+1)*CHUNK]
                    torch.addmm(lc_n, h[i0:i1], Rc.t(), beta=0, alpha=1, out=lc_n)
                    torch.addmm(lc_n, t[i0:i1], Uc.t(), beta=1, alpha=model.emb.alpha / model.emb.rank, out=lc_n)
                    yc = torch.where((y[i0:i1] >= c*CHUNK) & (y[i0:i1] < (c+1)*CHUNK), y[i0:i1] - c*CHUNK, -100)
                    valid = yc >= 0
                    nv = valid.sum().item()
                    if nv == 0:
                        continue
                    tot_s += F.cross_entropy(lc_n.float(), yc.long(), reduction='sum').item()
                    tot_v += nv
    model.train()
    return math.exp(tot_s / tot_v)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
lc_buf = torch.empty(2048, CHUNK, device=dev)
t0 = time.time()
for st in range(START, STEPS + 1):
    idx = torch.randint(0, tr_x.shape[0], (B,))
    bx, by = tr_x[idx].to(dev), tr_y[idx].to(dev)
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if st % 100 == 0:
        el = time.time() - t0
        print(f'  st={st:5d} | loss={loss:.3f} | tok/s={(st + 1) * B * SEQ / el:.0f} | {el:.0f}s', flush=True)
    if st > 0 and st % 2000 == 0:
        ppl = eval_ppl()
        best = min(best, ppl)
        print(f'  >>> EVAL st={st:5d} | val_ppl={ppl:.2f} | best={best:.2f}', flush=True)
        torch.save({'step': st, 'model_state': model.state_dict(), 'best_ppl': best},
                   os.path.join(OUT, 'ckpt_%05d.pt' % st))
ppl = eval_ppl(); best = min(best, ppl)
dur = time.time() - t0
vram = torch.cuda.max_memory_allocated() / 1e9
print(f'DONE: final_ppl={ppl:.2f} best={best:.2f} | {dur:.0f}s | peak {vram:.2f}GB', flush=True)
