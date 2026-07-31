"""
Four approaches to beat Transformers with V3 compression:
  A: Orthogonal R (structured random, norm-preserving)
  B: VQ bottleneck + V3-Embed on discrete codes
  C: Progressive V3 (first K layers full-rank, rest V3)
  D: Permutation R (sparse, dimension-reordering, lossless)
"""
import sys, os, math, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from condenser.v3_embed import V3WeightEmbedding
from condenser.baseline_transformer import BaselineAttention, BaselineTransformerLayer


# ===== SHARED UTILITY: V3 linear with gradient flow through R =====
class V3LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, R, U, V, alpha, rank):
        scaling = alpha / rank
        W = R + scaling * (U @ V.t())
        out = F.linear(x, W)
        ctx.save_for_backward(x, R, U, V)
        ctx.scaling = scaling
        return out
    @staticmethod
    def backward(ctx, grad_output):
        x, R, U, V = ctx.saved_tensors
        in_f = x.shape[-1]; out_f = grad_output.shape[-1]
        x_f = x.reshape(-1, in_f); go_f = grad_output.reshape(-1, out_f)
        grad_W = go_f.t() @ x_f
        s = ctx.scaling
        W_t = R.t() + s * (V @ U.t())
        return (F.linear(grad_output, W_t), None, s * (grad_W @ V), s * (grad_W.t() @ U), None, None)


def make_orthogonal_R(shape, device, dtype, seed):
    h = hashlib.md5(str(seed).encode()).hexdigest()
    seed_int = int(h[:8], 16)
    gen = torch.Generator(device='cpu').manual_seed(seed_int)
    A = torch.randn(*shape, generator=gen, device='cpu', dtype=dtype)
    if shape[0] >= shape[1]:
        Q, _ = torch.linalg.qr(A)
        return Q.to(device)
    Q, _ = torch.linalg.qr(A.t())
    return Q.t().to(device)


# ============================================================
# APPROACH A: V3 with Orthogonal R (norm-preserving)
# ============================================================
class V3OrthoLayer(nn.Module):
    def __init__(self, dim, heads, rank=16, alpha=64, seed=42, lid=0):
        super().__init__()
        self.heads = heads; self.dh = dim // heads; self.alpha = alpha; self.rank = rank
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for idx, (nm, od, id_) in enumerate([('q',dim,dim),('k',dim,dim),('v',dim,dim),('o',dim,dim),
                                              ('ff1',4*dim,dim),('ff2',dim,4*dim)]):
            s = seed + lid*100 + idx
            R = make_orthogonal_R((od, id_), dev, torch.float32, s)
            self.register_buffer(f'R_{nm}', R)
            setattr(self, f'U_{nm}', nn.Parameter(torch.randn(od, rank) * 0.01))
            setattr(self, f'V_{nm}', nn.Parameter(torch.randn(id_, rank) * 0.01))
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)

    def _l(self, x, nm):
        return V3LinearFn.apply(x, getattr(self, f'R_{nm}'), getattr(self, f'U_{nm}'),
                                getattr(self, f'V_{nm}'), self.alpha, self.rank)

    def forward(self, x):
        B,T,D = x.shape; h = self.dh
        n = self.norm1(x)
        Q = self._l(n,'q').view(B,T,self.heads,h).transpose(1,2)
        K = self._l(n,'k').view(B,T,self.heads,h).transpose(1,2)
        V = self._l(n,'v').view(B,T,self.heads,h).transpose(1,2)
        a = F.softmax(Q @ K.transpose(-2,-1) / math.sqrt(h), dim=-1)
        o = (a @ V).transpose(1,2).reshape(B,T,D)
        x = x + self._l(o,'o')
        n2 = self.norm2(x)
        x = x + self._l(F.relu(self._l(n2,'ff1')),'ff2')
        return x


class V3Ortho(nn.Module):
    def __init__(self, vs, D, L, H, r=16, a=64, s=42):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers = nn.ModuleList([V3OrthoLayer(D,H,r,a,s,i) for i in range(L)])
        self.norm = nn.LayerNorm(D)
    def forward(self, x):
        B,T=x.shape
        h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.emb.forward_head(self.norm(h))
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# APPROACH B: VQ + V3-Embed (discrete bottleneck)
# ============================================================
class VQV3Layer(nn.Module):
    """VQ quantize → V3-Embed decode → standard TF block"""
    def __init__(self, dim, heads, codebook_size=256, rank=16, alpha=64):
        super().__init__()
        self.codebook = V3WeightEmbedding(codebook_size, dim, rank, alpha)
        self.tf = BaselineTransformerLayer(dim, heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B,T,D = x.shape
        # Quantize: find nearest codebook entry per token
        R = self.codebook.R
        lora = (self.codebook.alpha/self.codebook.rank) * (self.codebook.U @ self.codebook.V.t())
        codes = R + lora  # (C, D)
        x_f = x.reshape(-1, D)
        dist = torch.cdist(x_f, codes)
        idx = dist.argmin(dim=-1).reshape(B, T)
        # Decode via V3-Embed
        z = self.codebook.forward_embed(idx)
        # TF block (straight-through: residual = x + tf_out - z + z.detach())
        out = self.tf(self.norm(z))
        # Straight-through estimator
        x = x + out + (z - z.detach())
        return x


class VQV3(nn.Module):
    def __init__(self, vs, D, L, H, cs=256, r=16, a=64):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers = nn.ModuleList([VQV3Layer(D,H,cs,r,a) for _ in range(L)])
        self.norm = nn.LayerNorm(D)
    def forward(self, x):
        B,T=x.shape
        h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.emb.forward_head(self.norm(h))
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# APPROACH D: V3 with Permutation R (lossless dimension shuffle)
# ============================================================
class V3PermLayer(nn.Module):
    """R is a random permutation matrix — reorders dimensions without mixing."""
    def __init__(self, dim, heads, rank=16, alpha=64, seed=42, lid=0):
        super().__init__()
        self.heads = heads; self.dh = dim // heads; self.alpha = alpha; self.rank = rank
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for idx, (nm, od, id_) in enumerate([('q',dim,dim),('k',dim,dim),('v',dim,dim),('o',dim,dim),
                                              ('ff1',4*dim,dim),('ff2',dim,4*dim)]):
            h = hashlib.md5(f"{seed+lid*100+idx}".encode()).hexdigest()
            gen = torch.Generator(device='cpu').manual_seed(int(h[:8],16))
            # Create permutation matrix: each output = one permuted input dimension
            perm = torch.randperm(id_, generator=gen)
            R = torch.zeros(od, id_)
            for oi in range(od):
                R[oi, perm[oi % id_]] = 1.0
            self.register_buffer(f'R_{nm}', R.to(dev, dtype=torch.float32))
            setattr(self, f'U_{nm}', nn.Parameter(torch.randn(od, rank) * 0.01))
            setattr(self, f'V_{nm}', nn.Parameter(torch.randn(id_, rank) * 0.01))
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)

    def _l(self, x, nm):
        return V3LinearFn.apply(x, getattr(self, f'R_{nm}'), getattr(self, f'U_{nm}'),
                                getattr(self, f'V_{nm}'), self.alpha, self.rank)

    def forward(self, x):
        B,T,D = x.shape; h = self.dh
        n = self.norm1(x)
        Q = self._l(n,'q').view(B,T,self.heads,h).transpose(1,2)
        K = self._l(n,'k').view(B,T,self.heads,h).transpose(1,2)
        V = self._l(n,'v').view(B,T,self.heads,h).transpose(1,2)
        a = F.softmax(Q @ K.transpose(-2,-1) / math.sqrt(h), dim=-1)
        o = (a @ V).transpose(1,2).reshape(B,T,D)
        x = x + self._l(o,'o')
        n2 = self.norm2(x)
        x = x + self._l(F.relu(self._l(n2,'ff1')),'ff2')
        return x


class V3Perm(nn.Module):
    def __init__(self, vs, D, L, H, r=16, a=64, s=42):
        super().__init__()
        self.emb=V3WeightEmbedding(vs,D,r,a); self.pos=nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers=nn.ModuleList([V3PermLayer(D,H,r,a,s,i) for i in range(L)])
        self.norm=nn.LayerNorm(D)
    def forward(self,x):
        B,T=x.shape; h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.emb.forward_head(self.norm(h))
    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# APPROACH C: Progressive V3 (first K full-rank, rest V3)
# ============================================================
class ProgressiveV3(nn.Module):
    def __init__(self, vs, D, L, H, num_std=1, r=16, a=64, s=42):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers = nn.ModuleList()
        for i in range(L):
            if i < num_std:
                self.layers.append(BaselineTransformerLayer(D, H))
            else:
                self.layers.append(V3OrthoLayer(D, H, r, a, s, i))
        self.norm = nn.LayerNorm(D)
    def forward(self, x):
        B,T=x.shape
        h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.emb.forward_head(self.norm(h))
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# BASELINE: V3-Embed + standard TF layers
# ============================================================
class Baseline(nn.Module):
    def __init__(self, vs, D, L, H, r=16, a=64):
        super().__init__()
        self.emb=V3WeightEmbedding(vs,D,r,a); self.pos=nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers=nn.ModuleList([BaselineTransformerLayer(D,H) for _ in range(L)])
        self.norm=nn.LayerNorm(D)
    def forward(self,x):
        B,T=x.shape; h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.emb.forward_head(self.norm(h))
    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===== DATA & TRAINING =====
def data(NUM=100, dev='cuda'):
    V=NUM+2; P=NUM; E=NUM+1
    tr=torch.tensor([[a,P,b,E,(a+b)%NUM] for a in range(50) for b in range(NUM)],dtype=torch.long,device=dev)
    te=torch.tensor([[a,P,b,E,(a+b)%NUM] for a in range(50,NUM) for b in range(NUM)],dtype=torch.long,device=dev)
    return V,tr,te

def acc(m,d,B=256):
    m.eval(); c=0; t=0
    with torch.no_grad():
        for i in range(0,d.shape[0],B):
            b=d[i:i+B]; p=m(b[:,:4]).argmax(dim=-1)
            c+=(p[:,-1]==b[:,-1]).sum().item(); t+=b.shape[0]
    return c/t*100


if __name__ == "__main__":
    torch.manual_seed(42)
    from condenser.genomatrix import _R_POOL
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}', flush=True)

    D=192; H=6; R=16; A=64; B=64; S=5000
    V, tr, te = data(100, dev)
    print(f'Train: {tr.shape[0]}, Test: {te.shape[0]}, Vocab: {V}\n', flush=True)

    configs = [
        ('0-Baseline V3-Embed+TF L=2', Baseline(V,D,2,H,R,A)),
        ('B-VQ+V3 L=2', VQV3(V,D,2,H,256,R,A)),
        ('C1-Prog 1Std+1V3 L=2', ProgressiveV3(V,D,2,H,1,R,A)),
        ('C2-Prog 1Std+2V3 L=3', ProgressiveV3(V,D,3,H,1,R,A)),
        ('C3-Prog 1Std+5V3 L=6', ProgressiveV3(V,D,6,H,1,R,A)),
        ('D1-PermR V3 L=2', V3Perm(V,D,2,H,R,A)),
        ('D2-PermR V3 L=6', V3Perm(V,D,6,H,R,A)),
        ('C4-Prog 2Std+4PermV3 L=6', ProgressiveV3(V,D,6,H,2,R,A)),
    ]
    # Override C4 to use Perm V3 layers instead of Ortho V3
    configs[-1][1].layers = nn.ModuleList(
        [BaselineTransformerLayer(D,H) for _ in range(2)] +
        [V3PermLayer(D,H,R,A,42,i) for i in range(4)]
    )

    print(f'{"Model":<30} {"Params":>10}')
    for n,m in configs:
        print(f'  {n:<28} {m.count_params():>10,}')
    print(flush=True)

    results = {}
    for name, model in configs:
        _R_POOL.clear()
        torch.cuda.reset_peak_memory_stats(dev)
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        best_ood = 0
        print(f'{"="*55}', flush=True)
        print(f'  {name}', flush=True)
        print(f'{"="*55}', flush=True)
        for step in range(S):
            model.train()
            idx = torch.randint(0, tr.shape[0], (B,))
            batch = tr[idx]
            loss = F.cross_entropy(model(batch[:,:4]).reshape(-1,V), batch[:,1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % 2500 == 0:
                o=acc(model,te); best_ood=max(best_ood,o)
                print(f'  step={step:>4} | train={acc(model,tr):.1f}% | ood={o:.1f}%', flush=True)
        o=acc(model,te); best_ood=max(best_ood,o)
        results[name] = {'train': acc(model,tr), 'ood': o, 'best_ood': best_ood, 'params': model.count_params()}
        print(f'  FINAL | train={results[name]["train"]:.1f}% | ood={o:.1f}% | best_ood={best_ood:.1f}%', flush=True)

    print(f'\n{"="*70}')
    print(f'  FINAL: 4 Approaches vs Baseline')
    print(f'{"="*70}')
    print(f'  {"Model":<30} {"Params":>8} {"Train":>7} {"OOD":>7} {"BestOOD":>8}')
    print(f'  {"-"*62}')
    for name, r in sorted(results.items()):
        print(f'  {name:<30} {r["params"]:>8,} {r["train"]:>6.1f}% {r["ood"]:>6.1f}% {r["best_ood"]:>7.1f}%')
