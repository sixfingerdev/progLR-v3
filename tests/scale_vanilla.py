"""
VANILLA TRANSFORMER BASELINE: standard nn.Embedding + nn.Linear head, full autograd,
NO analytic chunked head, NO checkpointing, NO CPU-data tricks. D=512, L=8, V=50K, WT103.
Answers: how fast is a REAL vanilla transformer at equal-param size on the 6GB card?
B=8 so the dense 50K-class logits (820MB) fit without paging.
"""
import os, sys, math, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.manual_seed(7)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'condenser', 'wt103_data.pkl')

from scale_compare_train import StdLayer

class Vanilla(nn.Module):
    def __init__(self, vs, D, H, L):
        super().__init__()
        self.emb = nn.Embedding(vs, D)
        self.pos = nn.Parameter(torch.randn(1, 1024, D) * 0.01)
        self.layers = nn.ModuleList([StdLayer(D, H) for _ in range(L)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, vs, bias=False)
    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos[:, :T]
        for l in self.layers: h = l(h)
        h = self.norm(h)
        return self.head(h)

D, H, L, V, B, SEQ = 512, 8, 8, 50000, int(os.environ.get('SCALE_B', '8')), 512
STEPS = int(os.environ.get('SCALE_STEPS', '100'))

def main():
    print('Loading data...', flush=True)
    with open(DATA, 'rb') as f: d = pickle.load(f)
    vs = d['vocab_size']
    dev = torch.device('cuda')
    tr_x = torch.tensor(d['tr_x'], dtype=torch.int32)
    tr_y = torch.tensor(d['tr_y'], dtype=torch.int32)
    model = Vanilla(vs, D, H, L).to(dev)
    pc = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n=== VANILLA Transformer (nn.Embedding + dense nn.Linear head): {pc:,} params ===', flush=True)
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()
    for st in range(STEPS):
        model.train()
        idx = torch.randint(0, tr_x.shape[0], (B,))
        bx, by = tr_x[idx].to(dev), tr_y[idx].to(dev)
        logits = model(bx)
        loss = F.cross_entropy(logits.reshape(-1, vs), by.reshape(-1).long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if st % 20 == 0:
            el = time.time() - t0
            print(f'  st={st:4d} | loss={loss.item():.3f} | tok/s={(st + 1) * B * SEQ / el:.0f} | {el:.0f}s', flush=True)
    dur = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f'  DONE: {dur:.0f}s | {(STEPS * B * SEQ) / dur:.0f} tok/s | peak {vram:.2f}GB', flush=True)

if __name__ == '__main__':
    main()
