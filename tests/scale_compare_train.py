"""
SCALE COMPARISON: ProgLR (2Std+6LR r=16) vs dense TF L=8, D=1024, V=50K, WT103.
Same embedding (V3WeightEmbedding), same analytic chunked head, same data, same seed, same steps.
Question: does 'same quality at fewer params' hold at real scale?

Scale note: D=1536/L=8 dense (229M, params+grads+Adam floor 3.66GB) paged on the 6GB card
(measured 5.83GB peak, 607 tok/s vs ~3.3K compute-bound) — dense does not fit consumer VRAM,
which is exactly the project thesis. D=1024/L=8 keeps a real-scale comparison that completes.

Speed notes (learned from final_report_train_v2.py + finetune_vocab250k.py):
- LR layers use a FACTORIZED forward (y = s*(x@V)@U.t()) -> same math, no full-rank W materialized
  (the old LRLinearFn built W = U@V.t() and did F.linear -> full-rank GEMM cost at D=1536).
- Chunked head: reused lc_buf, addmm(out=), in-place softmax (no temporaries per sub-batch).
- ALL layers gradient-checkpointed: at D=1536/L=8 saved activations (attn + FF1 hidden,
  ~450MB/layer) exceeded 6GB even for the 61.7M model -> CUDA paged (measured 7.5GB peak,
  6.2s/step). Checkpointing cuts activation memory to ~50MB/layer.
- Data tensors stay on CPU (int32, 811MB total) and batches move per-step: the dense run's
  floor is params+grads+Adam = 3.66GB and paged at 6.64GB peak otherwise.
- FP32 + TF32 ('high'), cuDNN benchmark on — same as the v2/finetune reference runs.
"""
import os, sys, math, pickle, time
import contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.manual_seed(7)

FP16 = os.environ.get('SCALE_FP16', '0') == '1'
NOCKPT = os.environ.get('SCALE_NOCKPT', '0') == '1'
HALF_CTX = torch.autocast('cuda', dtype=torch.float16) if FP16 else contextlib.nullcontext()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('SCALE_DATA', os.path.join(BASE, 'condenser', 'wt103_data.pkl'))
OUT = os.path.join(BASE, 'report_out_scale')
os.makedirs(OUT, exist_ok=True)

# === Low-Rank Layer (factorized, exact same math as LRLinearFn but no full W) ===
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

from condenser.v3_embed import V3WeightEmbedding

class Model(nn.Module):
    def __init__(self, vs, D, H, n_std, n_lr, r=16, a=64, ckpt_all=True):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1, 1024, D) * 0.01)
        self.ckpt_all = ckpt_all
        self.layers = nn.ModuleList()
        for _ in range(n_std): self.layers.append(StdLayer(D, H))
        for _ in range(n_lr): self.layers.append(LRLayer(D, H, r, a))
        self.norm = nn.LayerNorm(D)
    def get_hidden(self, x):
        B, T = x.shape
        h = self.emb.forward_embed(x) + self.pos[:, :T]
        for l in self.layers:
            if self.ckpt_all:
                h = ckpt.checkpoint(l, h, use_reentrant=False)
            else:
                h = l(h)
        return self.norm(h)

CHUNK = 50000
def head_loss_fn(h, U, V, R, y, alpha, rank, lc_buf, sub_batch, half=False, nce_k=0):
    """Analytic chunked head: loss value (detached) + exact dH, dU, dV.
    l_ic = h_i.R_c + s*(h_i.V).U_c, L = sum CE. In-place softmax on lc_buf.
    half=True: matmuls run on fp16 tensor cores (R/U/V cast per call), grads accumulate fp32.
    nce_k>0: NCE head — K negative samples per token (full target logits + sampled negatives),
    exact CE replaced by consistent NCE estimator. Chunks collapse to one (sampled subset)."""
    s = alpha / rank
    N, D = h.shape
    dH = torch.zeros_like(h); dU = torch.zeros_like(U); dV = torch.zeros_like(V)
    hf = h.half() if half else h
    tf = hf @ (V.half() if half else V)
    total = 0.0
    y = y.reshape(-1)
    if nce_k > 0:
        with torch.no_grad():
            neg = torch.randint(0, R.shape[0], (nce_k,), device=dev)  # shared negative set (K,)
        Rg = R[neg].float(); Ug = U[neg].float(); Ry = R[y].float(); Uy = U[y].float()
        hf32 = hf.float(); tf32 = tf.float()
        Vf = V.float() if half else V
        lt = torch.einsum('nd,nd->n', hf32, Ry) + s * torch.einsum('nd,nd->n', tf32, Uy)
        lc = torch.cat([lt.unsqueeze(1), hf32 @ Rg.t() + s * (tf32 @ Ug.t())], dim=1)
        total += F.cross_entropy(lc, torch.zeros(N, dtype=torch.long, device=dev),
                                 reduction='sum').item()
        torch.softmax(lc, dim=-1, out=lc)
        g = lc
        g[:, 0] -= 1.0
        dH.add_(g[:, 1:] @ Rg)
        dH.add_(s * (g[:, 1:] @ Ug) @ Vf.t())
        dH.add_(g[:, 0:1] * Ry)
        dH.add_(s * (g[:, 0:1] * Uy) @ Vf.t())
        dU.index_add_(0, y, s * g[:, 0:1] * tf32)
        dU.index_add_(0, neg, s * g[:, 1:].t() @ tf32)
        dV.add_(s * (hf32.t() @ (g[:, 1:] @ Ug + g[:, 0:1] * Uy)))
        return total, dH, dU, dV
        return total, dH, dU, dV
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
                if half:
                    Rc = Rc.half(); Uc = Uc.half()
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
                dV.add_((s * (h_b.t() @ (g @ Uc))).float())
    return total, dH, dU, dV

print('Loading data...', flush=True)
with open(DATA, 'rb') as f: d = pickle.load(f)
VS = d['vocab_size']
dev = torch.device('cuda')
tr_x = torch.tensor(d['tr_x'], dtype=torch.int32)
tr_y = torch.tensor(d['tr_y'], dtype=torch.int32)
te_x = torch.tensor(d['te_x'], dtype=torch.int32)
te_y = torch.tensor(d['te_y'], dtype=torch.int32)
del d

D, H, L, R, A, SEQ = (int(os.environ.get('SCALE_D', '1024')), int(os.environ.get('SCALE_H', '16')),
                       8, 16, 64, 512)
B, STEPS = int(os.environ.get('SCALE_B', '16')), int(os.environ.get('SCALE_STEPS', '15000'))
NCE_K = int(os.environ.get('SCALE_NCE', '0'))

_nstd, _nlr = os.environ.get('SCALE_NSTD'), os.environ.get('SCALE_NLR')
if _nstd is not None and _nlr is not None:
    configs = [(f'ProgLR {_nstd}Std+{_nlr}LR r={R} D={D}', Model(VS, D, H, int(_nstd), int(_nlr), R, A, ckpt_all=not NOCKPT))]
elif _nstd == 'dense':
    configs = [('Dense TF L=8 D=%d' % D, Model(VS, D, H, 8, 0, R, A, ckpt_all=not NOCKPT))]
else:
    configs = [
        ('ProgLR 2Std+6LR r=16', Model(VS, D, H, 2, 6, R, A, ckpt_all=not NOCKPT)),
        ('Dense TF L=8', Model(VS, D, H, 8, 0, R, A, ckpt_all=not NOCKPT)),
    ]

def eval_ppl(model, xs, ys, dim=None):
    model.eval(); tot_s = 0.0; tot_v = 0
    if dim is None: dim = model.pos.shape[-1]
    half = model.emb.R.dtype == torch.float16
    lc = torch.empty(1024, CHUNK, device=dev, dtype=torch.float16 if half else torch.float32)
    with torch.no_grad():
        for i in range(0, xs.shape[0], 8):
            bx = xs[i:i + 8].to(dev); by = ys[i:i + 8].to(dev)
            h = model.get_hidden(bx).reshape(-1, dim)
            hf = h.half() if half else h
            s = model.emb.alpha / model.emb.rank
            t = hf @ (model.emb.V.half() if half else model.emb.V)
            y = by.reshape(-1)
            n = hf.shape[0]
            for i0 in range(0, n, 1024):
                i1 = min(n, i0 + 1024); nb = i1 - i0
                lc_n = lc[:nb]
                for c in range(math.ceil(VS / CHUNK)):
                    Rc = model.emb.R[c*CHUNK:(c+1)*CHUNK]; Uc = model.emb.U[c*CHUNK:(c+1)*CHUNK]
                    if half:
                        Rc = Rc.half(); Uc = Uc.half()
                    torch.addmm(lc_n, hf[i0:i1], Rc.t(), beta=0, alpha=1, out=lc_n)
                    torch.addmm(lc_n, t[i0:i1], Uc.t(), beta=1, alpha=s, out=lc_n)
                    yc = torch.where((y[i0:i1] >= c*CHUNK) & (y[i0:i1] < (c+1)*CHUNK), y[i0:i1] - c*CHUNK, -100)
                    valid = yc >= 0
                    nv = valid.sum().item()
                    if nv > 0:
                        tot_s += F.cross_entropy(lc_n.float(), yc.long(), reduction='sum').item()
                        tot_v += nv
    return math.exp(tot_s / max(tot_v, 1))

def main():
    results = {}
    for name, model in configs:
        model.to(dev)
        if FP16:
            model.emb.R.data = model.emb.R.data.half()
        pc = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\n=== {name}: {pc:,} params ===', flush=True)
        torch.cuda.reset_peak_memory_stats()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        sub = int(os.environ.get('SCALE_SUBBATCH', '2048'))
        lc_buf = torch.empty(sub, CHUNK, device=dev, dtype=torch.float16 if FP16 else torch.float32)
        best = float('inf'); t0 = time.time()
        for st in range(STEPS):
            model.train()
            idx = torch.randint(0, tr_x.shape[0], (B,))
            bx, by = tr_x[idx].to(dev), tr_y[idx].to(dev)
            with HALF_CTX:
                h = model.get_hidden(bx)
            tot_s, dH, dU, dV = head_loss_fn(h.reshape(-1, D), model.emb.U, model.emb.V,
                                             model.emb.R, by, model.emb.alpha, model.emb.rank,
                                             lc_buf, sub, half=FP16, nce_k=NCE_K)
            loss = tot_s / (B * SEQ)
            with HALF_CTX:
                h.reshape(-1, D).backward(dH)
            with torch.no_grad():
                if model.emb.U.grad is None: model.emb.U.grad = dU
                else: model.emb.U.grad.add_(dU)
                if model.emb.V.grad is None: model.emb.V.grad = dV
                else: model.emb.V.grad.add_(dV)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            if st % int(os.environ.get('SCALE_LOG_EVERY', '100')) == 0:
                el = time.time() - t0
                print(f'  st={st:5d} | loss={loss:.3f} | tok/s={(st + 1) * B * SEQ / el:.0f} | {el:.0f}s', flush=True)
                if st > 0 and st % 2000 == 0:
                    ppl = eval_ppl(model, te_x, te_y)
                    best = min(best, ppl)
                    print(f'  >>> EVAL st={st:5d} | val_ppl={ppl:.2f} | best={best:.2f}', flush=True)
                    torch.save({'step': st, 'model_state': model.state_dict(), 'best_ppl': best},
                               os.path.join(OUT, f'best_{name.split()[0]}.pt'))
        ppl = eval_ppl(model, te_x, te_y); best = min(best, ppl)
        dur = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9
        results[name] = {'ppl': ppl, 'best': best, 'params': pc, 'sec': dur, 'vram': vram}
        print(f'  DONE {name}: final_ppl={ppl:.2f} best={best:.2f} | {dur:.0f}s | peak {vram:.2f}GB', flush=True)

    print('\n=== FINAL ===', flush=True)
    for name, r in results.items():
        print(f'{name:26} {r["params"]:>9,} params | best_ppl={r["best"]:.2f} final={r["ppl"]:.2f} | {r["sec"]:.0f}s | {r["vram"]:.2f}GB', flush=True)

if __name__ == '__main__':
    main()
