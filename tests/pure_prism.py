"""Pure PRISM (no LoRA) at D=512, L=16 on a+b mod 100."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; import torch.nn.functional as F
from condenser.v3_all import V3AllTransformer
device = torch.device('cuda')
NUM=100; PLUS=NUM; EQ=NUM+1; VOC=NUM+2
pairs=[(a,b) for a in range(NUM) for b in range(NUM)]
sums=[(a+b)%NUM for a,b in pairs]
import random; random.seed(42); idx=list(range(len(pairs))); random.shuffle(idx); sp=int(len(idx)*0.8); ti,te=idx[:sp],idx[sp:]
def mk(d): return torch.tensor([[pairs[i][0],PLUS,pairs[i][1],EQ,sums[i]] for i in d],dtype=torch.long,device=device)
tr,ts=mk(ti),mk(te)
def acc(model,d):
 model.eval();c=0;n=0
 with torch.no_grad():
  for i in range(0,d.shape[0],512):
   b=d[i:i+512];p=model(b[:,:4]).argmax(-1);c+=(p[:,-1]==b[:,-1]).sum().item();n+=b.shape[0]
 return c/n*100

log=open(r'C:\Users\enesa\law\tests\prism_log.txt','w')

# Pure PRISM: no LoRA, just V3-Embed + Anchor + PRISM modulation
class PurePRISMLayer(torch.nn.Module):
    def __init__(self, D, H, rank=16, base_seed=0):
        super().__init__()
        self.D=D; self.H=H; self.hd=D//H; self.rank=rank; self.seed=base_seed*100+42
        self.norm1=torch.nn.LayerNorm(D); self.norm2=torch.nn.LayerNorm(D)
    def forward(self, x, mod):
        B,T,D=x.shape; rs=x.device
        g=torch.Generator(device=rs)
        g.manual_seed(abs(self.seed)%(2**31-1))
        uv=torch.randn(12,D,self.rank,generator=g,device=rs,dtype=x.dtype)*(1/math.sqrt(self.rank))
        def W(i): return (uv[i*2+1]*mod[i].squeeze(0))@uv[i*2].T
        r1=x; x=self.norm1(x)
        Q=(x@W(0).T).view(B,T,self.H,self.hd).transpose(1,2)
        K=(x@W(1).T).view(B,T,self.H,self.hd).transpose(1,2)
        V=(x@W(2).T).view(B,T,self.H,self.hd).transpose(1,2)
        a=(Q@K.transpose(-2,-1))/math.sqrt(self.hd)
        m=torch.triu(torch.full((T,T),float('-inf'),device=rs),diagonal=1)
        o=(F.softmax(a+m,dim=-1)@V).transpose(1,2).reshape(B,T,D)
        x=r1+o@W(3).T
        r2=x; x=self.norm2(x)
        h=F.relu(x@W(4).T)
        x=r2+h@W(5).T
        return x

class PurePRISM(torch.nn.Module):
    def __init__(self, V, D, L, A=0, rank=16, er=8):
        super().__init__()
        from condenser.v3_all import V3Embedding
        self.embed=V3Embedding(V,D,er)
        self.pos=torch.nn.Parameter(torch.randn(1,512,D)*0.01)
        from condenser.v3_all import StdLayer
        self.anchor=torch.nn.ModuleList([StdLayer(D,D//64) for _ in range(min(A,L))])
        nP=max(0,L-min(A,L))
        self.prism=torch.nn.ModuleList([PurePRISMLayer(D,D//64,rank,i) for i in range(nP)])
        self.mod=torch.nn.Embedding(nP,6*rank)
        torch.nn.init.zeros_(self.mod.weight)
        self.norm=torch.nn.LayerNorm(D)
        self._eff=D*D*2*nP
    def effective_params(self): return self._eff
    def forward(self, x):
        x=self.embed.forward_embed(x)+self.pos[:,:x.shape[1]]
        for l in self.anchor: x=l(x)
        for i,l in enumerate(self.prism):
            mod=self.mod.weight[i].view(6,-1)
            x=l(x,mod)
        x=self.norm(x)
        return self.embed.forward_head(x) if hasattr(self.embed,'forward_head') else self.embed.head(x)
    def count(self): return sum(p.numel() for p in self.parameters())

for name,m in [
    ('Pure PRISM D=512 L=16 A=2', PurePRISM(VOC,512,16,2,16,8)),
    ('V3-All 2Std+2LR r=8', V3AllTransformer(VOC,128,4,4,embed_rank=8,hidden_rank=8,num_std_layers=2)),
]:
    m.to(device); opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0.01)
    n=m.count() if hasattr(m,'count') else sum(p.numel() for p in m.parameters())
    log.write(f'\n--- {name} ({n:,} params) ---\nStep Loss OOD%\n'); log.flush()
    for step in range(8001):
        m.train(); b=tr[torch.randint(0,tr.shape[0],(64,))]
        l=F.cross_entropy(m(b[:,:4]).reshape(-1,VOC),b[:,1:].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step%2000==0:
            a=acc(m,ts); log.write(f'{step:5d} {l.item():.4f} {a:.1f}%\n'); log.flush()
            if a>90: break
    log.write(f'Final: {acc(m,ts):.1f}%\n'); log.flush()
log.close();print('Done')
