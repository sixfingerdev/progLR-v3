"""V3-PRISM-H at moderate scale: D=384, L=16 on a+b mod 100."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; import torch.nn.functional as F
from condenser.v3_prism import V3PRISM
from condenser.baseline_transformer import BaselineTransformer
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

for name,m in [
    ('V3-PRISM D=384 L=12', V3PRISM(VOC,384,12,0,rank=16,lora_rank=32,embed_rank=8)),
    ('Std TF D=384 L=4', BaselineTransformer(VOC,384,4,6)),
]:
    m.to(device); opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0.01)
    n=m.count_trained() if hasattr(m,'count_trained') else sum(p.numel() for p in m.parameters())
    e=m.effective_params() if hasattr(m,'effective_params') else n
    log.write(f'\n--- {name} ({n:,} trained, eff={e:,}) ---\nStep Loss OOD%\n'); log.flush()
    t0=__import__('time').time()
    for step in range(6001):
        m.train(); b=tr[torch.randint(0,tr.shape[0],(64,))]
        l=F.cross_entropy(m(b[:,:4]).reshape(-1,VOC),b[:,1:].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step%2000==0:
            a=acc(m,ts); elapsed=__import__('time').time()-t0
            log.write(f'{step:5d} {l.item():.4f} {a:.1f}% [{elapsed:.0f}s]\n'); log.flush()
            if a>90: break
    log.write(f'Final: {acc(m,ts):.1f}%\n'); log.flush()
log.close();print('Done')
