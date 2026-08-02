"""
EQUAL-PARAM DENSE BASELINE for the scale comparison: dense TF at D=512, L=8 (~26.5M params)
vs ProgLR 28.8M (D=1024, 2Std+6LR r=16). Same embedding, same analytic chunked head, same
data, same seed, same steps. Answers: at EQUAL params, does ProgLR match dense quality?
Reuses Model + head_loss_fn + data loading from scale_compare_train.py (import-safe: main
is guarded by __name__ == '__main__').
"""
import os, sys, math, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
from scale_compare_train import Model, head_loss_fn, CHUNK, DATA, dev, VS, tr_x, tr_y, te_x, te_y, eval_ppl

D, H, L, R, A, SEQ = 512, 8, 8, 16, 64, 512
B, STEPS = 16, int(os.environ.get('SCALE_STEPS', '10000'))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'report_out_scale')
os.makedirs(OUT, exist_ok=True)

def main():
    model = Model(VS, D, H, 8, 0, R, A)
    model.to(dev)
    pc = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n=== Dense TF L=8 D={D} (equal-param): {pc:,} params ===', flush=True)
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    lc_buf = torch.empty(2048, CHUNK, device=dev)
    best = float('inf'); t0 = time.time()
    for st in range(STEPS):
        model.train()
        idx = torch.randint(0, tr_x.shape[0], (B,))
        bx, by = tr_x[idx].to(dev), tr_y[idx].to(dev)
        h = model.get_hidden(bx)
        tot_s, dH, dU, dV = head_loss_fn(h.reshape(-1, D), model.emb.U, model.emb.V,
                                         model.emb.R, by, model.emb.alpha, model.emb.rank,
                                         lc_buf, 2048)
        loss = tot_s / (B * SEQ)
        h.reshape(-1, D).backward(dH)
        with torch.no_grad():
            if model.emb.U.grad is None: model.emb.U.grad = dU
            else: model.emb.U.grad.add_(dU)
            if model.emb.V.grad is None: model.emb.V.grad = dV
            else: model.emb.V.grad.add_(dV)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if st % 100 == 0:
            el = time.time() - t0
            print(f'  st={st:5d} | loss={loss:.3f} | tok/s={(st + 1) * B * SEQ / el:.0f} | {el:.0f}s', flush=True)
        if st > 0 and st % 2000 == 0:
            ppl = eval_ppl(model, te_x, te_y)
            best = min(best, ppl)
            print(f'  >>> EVAL st={st:5d} | val_ppl={ppl:.2f} | best={best:.2f}', flush=True)
            torch.save({'step': st, 'model_state': model.state_dict(), 'best_ppl': best},
                       os.path.join(OUT, 'best_DenseEq.pt'))
    ppl = eval_ppl(model, te_x, te_y); best = min(best, ppl)
    dur = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f'  DONE: final_ppl={ppl:.2f} best={best:.2f} | {dur:.0f}s | peak {vram:.2f}GB', flush=True)

if __name__ == '__main__':
    main()
