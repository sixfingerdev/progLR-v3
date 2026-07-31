"""Benchmark: original dense-head step vs v2 analytic-chunked-head step.
Also gradchecks v2's analytic dH/dU/dV against autograd reference.
Usage: python tests/bench_head.py [steps=20]
"""
import os, sys, math, pickle, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'condenser', 'wt103_data.pkl')
sys.path.insert(0, os.path.join(BASE, 'tests'))

import final_report_train_v2 as v2
from final_report_train_v2 import ProgLR, head_loss_fn, CHUNK

dev = torch.device('cuda')
print('Loading data...', flush=True)
with open(DATA, 'rb') as f:
    d = pickle.load(f)
VS = d['vocab_size']
tr_x = torch.tensor(d['tr_x'], device=dev, dtype=torch.int32)
tr_y = torch.tensor(d['tr_y'], device=dev, dtype=torch.int32)
del d

B, D, SEQ = 16, 768, 512
torch.manual_seed(0)
model = ProgLR(VS, D, 2, 12, 1, 16, 64).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# ---------- GRADCHECK: v2 analytic vs autograd reference ----------
print('\n--- Gradcheck (analytic vs autograd) ---', flush=True)

# reference: full autograd through model.forward_head (original math)
torch.manual_seed(123)
bx = tr_x[0:2]; by = tr_y[0:2]
h_ref = model.get_hidden(bx).reshape(-1, D)
out_ref = model.emb.forward_head(h_ref.reshape(2, SEQ, D))
loss_ref = F.cross_entropy(out_ref.reshape(-1, VS), by.reshape(-1).long(), reduction='sum')
loss_ref.backward()
g_ref_U = model.emb.U.grad.clone()
g_ref_V = model.emb.V.grad.clone()
g_ref_pos = model.pos.grad.clone()
model.zero_grad(set_to_none=True)

# analytic: same input, v2 head_loss_fn
bx2 = tr_x[0:2]; by2 = tr_y[0:2]
h2 = model.get_hidden(bx2).reshape(-1, D)
lc_buf = torch.empty(2048, CHUNK, device=dev)
tot, dH, dU, dV = head_loss_fn(h2, model.emb.U, model.emb.V, model.emb.R, by2,
                               model.emb.alpha, model.emb.rank, lc_buf, 2048)
h2.backward(dH)
with torch.no_grad():
    model.emb.U.grad.add_(dU)
    model.emb.V.grad.add_(dV)
g_an_U = model.emb.U.grad.clone()
g_an_V = model.emb.V.grad.clone()
g_an_pos = model.pos.grad.clone()
model.zero_grad(set_to_none=True)

print(f'loss_an={tot:.4f} loss_ref={loss_ref.item():.4f}')
def nerr(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)
errU = nerr(g_an_U, g_ref_U)
errV = nerr(g_an_V, g_ref_V)
errP = nerr(g_an_pos, g_ref_pos)
print(f'U grad: norm-err={errU:.2e}')
print(f'V grad: norm-err={errV:.2e}')
print(f'pos grad (through layers): norm-err={errP:.2e}')
ok = (errU < 1e-3 and errV < 1e-3 and errP < 1e-3)
print(f'GRADCHECK: {"PASS" if ok else "FAIL"}')
if not ok:
    sys.exit(1)

# ---------- BENCHMARK ----------
def bench(mode, steps):
    torch.manual_seed(0)
    for p in model.parameters():
        p.detach().zero_()
        p.detach().normal_(0, 0.02)
    if mode == 'orig':
        model.pos.detach().normal_(0, 0.01)
    for g in opt.state.values():
        for t in g.values():
            if isinstance(t, torch.Tensor): t.zero_()
    lc_buf = torch.empty(2048, CHUNK, device=dev)
    warm = 3
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for st in range(warm + steps):
        idx = torch.randint(0, tr_x.shape[0], (B,), device=dev)
        bx, by = tr_x[idx], tr_y[idx]
        if mode == 'orig':
            out = model(bx)
            loss = F.cross_entropy(out.reshape(-1, VS), by.reshape(-1).long())
            loss.backward()
        else:
            h = model.get_hidden(bx)
            tot, dH, dU, dV = head_loss_fn(h.reshape(-1, D), model.emb.U, model.emb.V,
                                           model.emb.R, by, model.emb.alpha, model.emb.rank,
                                           lc_buf, 2048)
            h.reshape(-1, D).backward(dH)
            with torch.no_grad():
                if model.emb.U.grad is None: model.emb.U.grad = dU
                else: model.emb.U.grad.add_(dU)
                if model.emb.V.grad is None: model.emb.V.grad = dV
                else: model.emb.V.grad.add_(dV)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return dt, steps * B * SEQ / dt

print('\n--- Benchmark ---', flush=True)
dt, tps = bench('orig', STEPS)
print(f'orig: {dt*1000:.0f} ms/step | {tps:,.0f} tok/s', flush=True)
peak_orig = torch.cuda.max_memory_allocated() / 1e9
torch.cuda.reset_peak_memory_stats()
dt2, tps2 = bench('v2', STEPS)
peak_v2 = torch.cuda.max_memory_allocated() / 1e9
print(f'v2  : {dt2*1000:.0f} ms/step | {tps2:,.0f} tok/s | speedup {dt/dt2:.2f}x', flush=True)
print(f'peak mem: orig={peak_orig:.2f} GB, v2={peak_v2:.2f} GB', flush=True)
