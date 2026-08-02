"""Verify ckpt_12000.pt: eval PPL must match training (70.70) + BPE generate test.
Self-contained: model classes copied here so importing does NOT re-run training.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt

torch.set_float32_matmul_precision('high')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from condenser.v3_embed import V3WeightEmbedding

D, H, R, A = 1024, 16, 16, 64
PATTERN = ['std', 'std'] + ['lr'] * 4 + ['std'] + ['lr'] * 4 + ['std']
VS = 50000
CHUNK = 50000

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
        B, T, D = x.shape; h = self.dh
        n = self.n1(x)
        Q = self.W_Q(n).view(B, T, self.h, h).transpose(1, 2)
        K = self.W_K(n).view(B, T, self.h, h).transpose(1, 2)
        V = self.W_V(n).view(B, T, self.h, h).transpose(1, 2)
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

dev = torch.device('cuda')

print('Loading ckpt_12000.pt...', flush=True)
ck = torch.load(os.path.join(BASE, 'report_out_sandwich_bpe', 'ckpt_12000.pt'),
                map_location=dev, weights_only=False)
print(f"step={ck['step']} best_ppl={ck['best_ppl']:.2f}", flush=True)

model = Model(VS, D, H, PATTERN, R, A).to(dev)
model.load_state_dict(ck['model_state'])
model.eval()
pc = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {pc:,}', flush=True)

print('Loading eval data...', flush=True)
import numpy as np
d = np.load(os.path.join(BASE, 'condenser', 'wt103_bpe.npz'))
te_x = torch.from_numpy(d['te_x'])
te_y = torch.from_numpy(d['te_y'])

print('Re-running eval PPL...', flush=True)
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
ppl = math.exp(tot_s / tot_v)
print(f'=== VERIFIED EVAL PPL: {ppl:.2f} (training said 70.70) ===', flush=True)

print('Loading BPE tokenizer...', flush=True)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(BASE, 'condenser', 'wt103_bpe_tokenizer.json'))

def sample_logits(logits, temp=0.8, top_k=50, top_p=0.9, rep_penalty=1.1, used_ids=None):
    logits = logits / temp
    if rep_penalty is not None and used_ids is not None:
        for tid in used_ids:
            if logits[tid] > 0:
                logits[tid] = logits[tid] / rep_penalty
            else:
                logits[tid] = logits[tid] * rep_penalty
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.numel())
        thresh = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < thresh, float('-inf'))
    if top_p is not None and top_p < 1.0:
        sorted_l, idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        mask = cum - sorted_l.softmax(dim=-1) > top_p
        sorted_l[mask] = float('-inf')
        logits = torch.gather(sorted_l, -1, idx.argsort(dim=-1))
    return torch.multinomial(F.softmax(logits, dim=-1), 1).item()

def generate(prompt, steps=60, temp=0.8, top_k=50, top_p=0.9, rep_penalty=1.1):
    ids = tok.encode(prompt.lower()).ids
    ctx = torch.tensor([ids], dtype=torch.long, device=dev)
    out = list(ids)
    used = ids[:]
    with torch.no_grad():
        for _ in range(steps):
            h = model.get_hidden(ctx[:, -512:])
            x = h[:, -1, :]
            logits = (x @ model.emb.R.t()) + (model.emb.alpha / model.emb.rank) * ((x @ model.emb.V) @ model.emb.U.t())
            nxt = sample_logits(logits[0], temp, top_k, top_p, rep_penalty, used[-128:])
            out.append(nxt)
            used.append(nxt)
            ctx = torch.cat([ctx, torch.tensor([[nxt]], device=dev)], dim=1)
    return tok.decode(out)

for prompt in ['the history of', 'in the beginning', 'the government announced', 'one of the most']:
    print(f'\n>>> "{prompt}" (top_k=50 top_p=0.9 rep=1.1 temp=0.8)', flush=True)
    print(generate(prompt), flush=True)
print('\nDONE', flush=True)
