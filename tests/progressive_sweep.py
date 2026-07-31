"""
Progressive V3: 1 standard layer + N V3 layers, find the sweet spot.
N = 1, 2, 3, 4, 5
"""
import sys, os, math, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from condenser.v3_embed import V3WeightEmbedding
from condenser.baseline_transformer import BaselineTransformerLayer

# === V3 Reusable Components ===
class V3LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, R, U, V, alpha, rank):
        s=alpha/rank; out=F.linear(x,R+s*(U@V.t()))
        ctx.save_for_backward(x,R,U,V); ctx.s=s; return out
    @staticmethod
    def backward(ctx, go):
        x,R,U,V=ctx.saved_tensors; i_f=x.shape[-1]; o_f=go.shape[-1]
        x_f=x.reshape(-1,i_f); go_f=go.reshape(-1,o_f); gW=go_f.t()@x_f; s=ctx.s
        return (F.linear(go,R.t()+s*(V@U.t())),None,s*(gW@V),s*(gW.t()@U),None,None)

class V3Layer(nn.Module):
    def __init__(self, dim, heads, rank=16, alpha=64, seed=42, lid=0):
        super().__init__()
        self.h=heads; self.dh=dim//heads; self.a=alpha; self.r=rank
        dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i,(n,od,id_) in enumerate([('q',dim,dim),('k',dim,dim),('v',dim,dim),('o',dim,dim),('ff1',4*dim,dim),('ff2',dim,4*dim)]):
            g=torch.Generator(device='cpu').manual_seed(int(hashlib.md5(f"{seed+lid*100+i}".encode()).hexdigest()[:8],16))
            R=torch.randn(od,id_,generator=g,device='cpu',dtype=torch.float32)/math.sqrt(id_)
            if od>=id_: Q,_=torch.linalg.qr(R)
            else: Q,_=torch.linalg.qr(R.t()); Q=Q.t()
            self.register_buffer(f'R_{n}',Q.to(dev))
            setattr(self,f'U_{n}',nn.Parameter(torch.randn(od,rank)*0.01))
            setattr(self,f'V_{n}',nn.Parameter(torch.randn(id_,rank)*0.01))
        self.n1=nn.LayerNorm(dim); self.n2=nn.LayerNorm(dim)
    def _l(self,x,n): return V3LinearFn.apply(x,getattr(self,f'R_{n}'),getattr(self,f'U_{n}'),getattr(self,f'V_{n}'),self.a,self.r)
    def forward(self,x):
        B,T,D=x.shape; h=self.dh; n=self.n1(x)
        Q=self._l(n,'q').view(B,T,self.h,h).transpose(1,2)
        K=self._l(n,'k').view(B,T,self.h,h).transpose(1,2)
        V=self._l(n,'v').view(B,T,self.h,h).transpose(1,2)
        a=F.softmax(Q@K.transpose(-2,-1)/math.sqrt(h),dim=-1)
        o=(a@V).transpose(1,2).reshape(B,T,D)
        x=x+self._l(o,'o'); n2=self.n2(x)
        x=x+self._l(F.relu(self._l(n2,'ff1')),'ff2')
        return x

class ProgV3(nn.Module):
    def __init__(self, vs, D, L, H, r=16, a=64, s=42):
        super().__init__()
        self.emb=V3WeightEmbedding(vs,D,r,a); self.pos=nn.Parameter(torch.randn(1,512,D)*0.01)
        self.layers=nn.ModuleList([BaselineTransformerLayer(D,H)]+[V3Layer(D,H,r,a,s,i) for i in range(L-1)])
        self.norm=nn.LayerNorm(D)
    def forward(self,x):
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
    D=192; H=6; R=16; A=64; B=64; S=10000
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V,tr,te=data(100,dev)
    print(f'Device: {dev}', flush=True)
    print(f'Train: {tr.shape[0]}, Test: {te.shape[0]}, Vocab: {V}\n', flush=True)

    cfgs=[('0-Baseline TF L=2',Baseline(V,D,2,H,R,A)),
          ('1-Prog 1Std+1V3 L=2',ProgV3(V,D,2,H,R,A)),
          ('2-Prog 1Std+2V3 L=3',ProgV3(V,D,3,H,R,A)),
          ('3-Prog 1Std+3V3 L=4',ProgV3(V,D,4,H,R,A)),
          ('4-Prog 1Std+4V3 L=5',ProgV3(V,D,5,H,R,A)),
          ('5-Prog 1Std+5V3 L=6',ProgV3(V,D,6,H,R,A)),
          ('6-Baseline TF L=6',Baseline(V,D,6,H,R,A))]

    print(f'{"Model":<30} {"Params":>10}')
    for n,m in cfgs: print(f'  {n:<28} {m.cp():>10,}')
    print(flush=True)

    from condenser.genomatrix import _R_POOL
    res={}
    for name,model in cfgs:
        _R_POOL.clear(); torch.cuda.reset_peak_memory_stats(dev)
        model.to(dev)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
        bo=0; print(f'{"="*55}',flush=True); print(f'  {name}',flush=True); print(f'{"="*55}',flush=True)
        for st in range(S):
            model.train()
            idx=torch.randint(0,tr.shape[0],(B,)); batch=tr[idx]
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
    print(f'  Progressive V3: Sweet Spot Search')
    print(f'{"="*70}')
    print(f'  {"Model":<30} {"Params":>8} {"Train":>7} {"OOD":>7} {"BestOOD":>8}')
    print(f'  {"-"*62}')
    for n,r in sorted(res.items()):
        print(f'  {n:<30} {r["params"]:>8,} {r["train"]:>6.1f}% {r["ood"]:>6.1f}% {r["best_ood"]:>7.1f}%')
