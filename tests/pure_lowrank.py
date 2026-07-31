"""
Pure Low-Rank Transformer: W = (alpha/rank) * U @ V^T
No random R. Every parameter is trained.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from condenser.v3_embed import V3WeightEmbedding
from condenser.baseline_transformer import BaselineTransformerLayer

class LowRankLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, U, V, alpha, rank):
        s = alpha / rank
        out = F.linear(x, s * (U @ V.t()))
        ctx.save_for_backward(x, U, V)
        ctx.s = s
        return out
    @staticmethod
    def backward(ctx, go):
        x, U, V = ctx.saved_tensors
        in_f, out_f = x.shape[-1], go.shape[-1]
        x_f = x.reshape(-1, in_f); go_f = go.reshape(-1, out_f)
        s = ctx.s
        gW = go_f.t() @ x_f
        return (F.linear(go, s * (V @ U.t())), s * (gW @ V), s * (gW.t() @ U), None, None)


class LowRankLayer(nn.Module):
    def __init__(self, dim, heads, rank=32, alpha=64):
        super().__init__()
        self.h, self.dh, self.r, self.a = heads, dim//heads, rank, alpha
        for nm, od, id_ in [('q',dim,dim),('k',dim,dim),('v',dim,dim),('o',dim,dim),
                             ('ff1',4*dim,dim),('ff2',dim,4*dim)]:
            setattr(self, f'U_{nm}', nn.Parameter(torch.randn(od, rank) * 0.02))
            setattr(self, f'V_{nm}', nn.Parameter(torch.randn(id_, rank) * 0.02))
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def _l(self, x, nm):
        return LowRankLinearFn.apply(x, getattr(self, f'U_{nm}'),
                                      getattr(self, f'V_{nm}'), self.a, self.r)

    def forward(self, x):
        B,T,D = x.shape; h = self.dh
        n = self.n1(x)
        Q = self._l(n,'q').view(B,T,self.h,h).transpose(1,2)
        K = self._l(n,'k').view(B,T,self.h,h).transpose(1,2)
        V = self._l(n,'v').view(B,T,self.h,h).transpose(1,2)
        a = F.softmax(Q @ K.transpose(-2,-1) / math.sqrt(h), dim=-1)
        o = (a @ V).transpose(1,2).reshape(B,T,D)
        x = x + self._l(o,'o')
        n2 = self.n2(x)
        x = x + self._l(F.relu(self._l(n2,'ff1')),'ff2')
        return x


class PureLowRank(nn.Module):
    def __init__(self, vs, D, L, H, r=32, a=64):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers = nn.ModuleList([LowRankLayer(D,H,r,a) for _ in range(L)])
        self.norm = nn.LayerNorm(D)
    def forward(self, x):
        B,T=x.shape; h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.emb.forward_head(self.norm(h))
    def cp(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ProgressiveLowRank(nn.Module):
    """First num_std layers = standard TF, rest = LowRank."""
    def __init__(self, vs, D, L, H, num_std=1, r=32, a=64):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers = nn.ModuleList()
        for i in range(L):
            if i < num_std:
                self.layers.append(BaselineTransformerLayer(D, H))
            else:
                self.layers.append(LowRankLayer(D,H,r,a))
        self.norm = nn.LayerNorm(D)
    def forward(self, x):
        B,T=x.shape; h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.emb.forward_head(self.norm(h))
    def cp(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


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


def data(N=100,dev='cuda'):
    V=N+2; P=N; E=N+1
    tr=torch.tensor([[a,P,b,E,(a+b)%N] for a in range(50) for b in range(N)],dtype=torch.long,device=dev)
    te=torch.tensor([[a,P,b,E,(a+b)%N] for a in range(50,N) for b in range(N)],dtype=torch.long,device=dev)
    return V,tr,te

def acc(m,d,B=256):
    m.eval(); c=0; t=0
    with torch.no_grad():
        for i in range(0,d.shape[0],B):
            b=d[i:i+B]; p=m(b[:,:4]).argmax(dim=-1)
            c+=(p[:,-1]==b[:,-1]).sum().item(); t+=b.shape[0]
    return c/t*100


if __name__=="__main__":
    D=192; H=6; BS=64; S=10000
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V,tr,te=data(100,dev)
    print(f'Device: {dev}',flush=True)
    print(f'Train: {tr.shape[0]}, Test: {te.shape[0]}, Vocab: {V}\n',flush=True)

    cfgs = [
        ('0-Baseline TF L=2', Baseline(V,D,2,H,16,64)),
        ('1-PureLR r=16 L=2', PureLowRank(V,D,2,H,16,64)),
        ('2-PureLR r=32 L=2', PureLowRank(V,D,2,H,32,64)),
        ('3-PureLR r=48 L=2', PureLowRank(V,D,2,H,48,64)),
        ('4-PureLR r=32 L=4', PureLowRank(V,D,2,H,32,64)),  # Wait, L=2 here twice, fix
        ('5-Prog 1Std+1LR r=32 L=2', ProgressiveLowRank(V,D,2,H,1,32,64)),
        ('6-Prog 1Std+2LR r=32 L=3', ProgressiveLowRank(V,D,3,H,1,32,64)),
        ('7-Baseline TF L=4', Baseline(V,D,4,H,16,64)),
    ]
    # Fix cfg 4
    cfgs[4] = ('4-PureLR r=32 L=4', PureLowRank(V,D,4,H,32,64))

    print(f'{"Model":<32} {"Params":>10} {"D*H":>8}')
    for n,m in cfgs:
        p=m.cp(); d=sum(1 for _ in m.parameters() if _.requires_grad)
        print(f'  {n:<30} {p:>10,} {0:>8}')
    print(flush=True)

    from condenser.genomatrix import _R_POOL
    res={}
    for name,model in cfgs:
        _R_POOL.clear(); torch.cuda.reset_peak_memory_stats(dev)
        model.to(dev)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
        bo=0
        print(f'{"="*55}',flush=True); print(f'  {name}',flush=True); print(f'{"="*55}',flush=True)
        for st in range(S):
            model.train()
            idx=torch.randint(0,tr.shape[0],(BS,)); batch=tr[idx]
            loss=F.cross_entropy(model(batch[:,:4]).reshape(-1,V),batch[:,1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if st%2500==0:
                o=acc(model,te); bo=max(bo,o)
                print(f'  step={st:>4} | train={acc(model,tr):.1f}% | ood={o:.1f}%',flush=True)
        o=acc(model,te); bo=max(bo,o)
        res[name]={'train':acc(model,tr),'ood':o,'best_ood':bo,'params':model.cp()}
        print(f'  FINAL | train={res[name]["train"]:.1f}% | ood={o:.1f}% | best={bo:.1f}%\n',flush=True)

    print(f'\n{"="*70}')
    print(f'  Pure Low-Rank Transformer Results')
    print(f'{"="*70}')
    print(f'  {"Model":<32} {"Params":>8} {"Train":>7} {"OOD":>7} {"BestOOD":>8}')
    print(f'  {"-"*62}')
    for n,r in sorted(res.items()):
        print(f'  {n:<32} {r["params"]:>8,} {r["train"]:>6.1f}% {r["ood"]:>6.1f}% {r["best_ood"]:>7.1f}%')
