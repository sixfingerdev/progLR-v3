"""
Quick OOD test: V3-PRISM with generated attention.
Logs to output file instead of stdout.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from condenser.v3_prism import V3PRISM

device = torch.device('cuda')
NUM = 100; PLUS = NUM; EQ = NUM + 1; VOC = NUM + 2

pairs = [(a, b) for a in range(NUM) for b in range(NUM)]
sums = [(a + b) % NUM for a, b in pairs]
import random; random.seed(42)
idx = list(range(len(pairs))); random.shuffle(idx)
sp = int(len(idx) * 0.8); ti, te = idx[:sp], idx[sp:]

def mk(d):
    return torch.tensor([[pairs[i][0], PLUS, pairs[i][1], EQ, sums[i]] for i in d],
                        dtype=torch.long, device=device)
tr, ts = mk(ti), mk(te)

def acc(model, d):
    model.eval(); c, n = 0, 0
    with torch.no_grad():
        for i in range(0, d.shape[0], 512):
            b = d[i:i+512]
            p = model(b[:, :4]).argmax(-1)
            c += (p[:, -1] == b[:, -1]).sum().item(); n += b.shape[0]
    return c/n*100

log = open(r'C:\Users\enesa\law\tests\prism_log.txt', 'w')

for cfg in [
    ('V3-PRISM 0A+4P r=16', V3PRISM(VOC, 128, 4, 0, 16, 8)),
    ('V3-PRISM 2A+2P r=16', V3PRISM(VOC, 128, 4, 2, 16, 8)),
]:
    name, model = cfg
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n = model.count_trained()
    e = model.effective_params()
    log.write(f'\n--- {name} ({n:,} params, eff={e:,}) ---\n')
    log.write(f'Step  Loss   OOD%\n')
    log.flush()
    for step in range(5001):
        model.train()
        b = tr[torch.randint(0, tr.shape[0], (128,))]
        l = F.cross_entropy(model(b[:, :4]).reshape(-1, VOC), b[:, 1:].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            a = acc(model, ts)
            log.write(f'{step:5d}  {l.item():.4f}  {a:.1f}%\n')
            log.flush()
            if a > 50:
                break
    log.write(f'Final: {acc(model, ts):.1f}% OOD\n')
    log.flush()
log.close()
print('Done. Check prism_log.txt')
