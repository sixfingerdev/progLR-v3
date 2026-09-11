"""Trainer: config-driven training loop with exact-CE or NCE head."""
import contextlib
import math
import os
import pickle
import time

import torch
import torch.nn as nn

from .config import TrainConfig
from .head import head_loss_fn, eval_ppl
from .model import ProgLR

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    tr_x = torch.tensor(d['tr_x'], dtype=torch.int32)
    tr_y = torch.tensor(d['tr_y'], dtype=torch.int32)
    te_x = torch.tensor(d['te_x'], dtype=torch.int32)
    te_y = torch.tensor(d['te_y'], dtype=torch.int32)
    vs = d['vocab_size']
    return vs, tr_x, tr_y, te_x, te_y


def make_model(vs, cfg: TrainConfig):
    return ProgLR(vs, cfg.d, cfg.h, cfg.n_std, cfg.n_lr, cfg.r, cfg.a,
                  max_pos=cfg.max_pos, checkpoint_all=cfg.checkpoint_all,
                  act=cfg.act, rope=cfg.rope, r_seed=cfg.r_seed)


def _head_at(schedule, st):
    """Active (head, nce_k) at step st for a head schedule."""
    h, k = schedule[0][1], schedule[0][2]
    for s, hh, kk in schedule:
        if st >= s:
            h, k = hh, kk
    return h, k


def _flog(*args, **kwargs):
    """Buffering-free default logger (pipe/tee-safe)."""
    print(*args, flush=True, **kwargs)


def _validate_cfg(cfg: TrainConfig):
    """Fail fast with clear errors instead of silently training something else."""
    if cfg.head not in ('exact', 'nce'):
        raise ValueError(f"head must be 'exact' or 'nce', got {cfg.head!r}")
    if not os.path.exists(cfg.data):
        raise FileNotFoundError(f"data file not found: {cfg.data}")
    if cfg.chunk <= 0 or cfg.sub_batch <= 0 or cfg.batch <= 0 or cfg.steps <= 0:
        raise ValueError(f'invalid size setting: chunk={cfg.chunk} sub_batch={cfg.sub_batch} '
                         f'batch={cfg.batch} steps={cfg.steps}')
    seen = -1
    for s, h, k in (cfg.head_schedule or []):
        if s < 0 or s > cfg.steps:
            raise ValueError(f'head_schedule step {s} out of range [0, {cfg.steps}]')
        if h not in ('exact', 'nce'):
            raise ValueError(f'head_schedule head must be "exact" or "nce", got {h!r}')
        if s < seen:
            raise ValueError(f'head_schedule steps must be ascending, got {s} after {seen}')
        seen = s


def train(cfg: TrainConfig, log=_flog, save_best=True):
    """Train one ProgLR model. Returns result dict (and leaves checkpoints in out_dir)."""
    _validate_cfg(cfg)
    cfg.resolve(_BASE)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    vs, tr_x, tr_y, te_x, te_y = load_data(cfg.data)
    dev = torch.device(cfg.device)
    model = make_model(vs, cfg)
    model.to(dev)
    if cfg.fp16:
        model.emb.R.data = model.emb.R.data.half()

    name = f'ProgLR {cfg.n_std}Std+{cfg.n_lr}LR r={cfg.r} D={cfg.d}'
    head_label = ('schedule' if cfg.head_schedule else cfg.head) + \
                 (f' (NCE K={cfg.nce_k})' if cfg.head == 'nce' else '')
    pc = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'\n=== {name}: {pc:,} params | vocab {vs:,} | steps {cfg.steps} | '
        f'head={head_label} ===')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lc_buf = torch.empty(cfg.sub_batch, cfg.chunk, device=dev,
                         dtype=torch.float16 if cfg.fp16 else torch.float32)
    half_ctx = torch.autocast('cuda', dtype=torch.float16) if cfg.fp16 else contextlib.nullcontext()
    schedule = cfg.head_schedule or [(0, cfg.head, cfg.nce_k)]
    cur_head, nce_k = _head_at(schedule, 0)

    best = float('inf'); t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def evaluate():
        model.eval()
        ppl = eval_ppl(model, te_x, te_y, chunk=cfg.chunk)
        model.train()
        return ppl

    for st in range(cfg.steps):
        h_type, k = _head_at(schedule, st)
        if h_type != cur_head or k != nce_k:
            cur_head, nce_k = h_type, k
            log(f'  >>> HEAD SWITCH st={st:5d} | {h_type}' + (f' (NCE K={k})' if h_type == 'nce' else '') + ' <<<')
        idx = torch.randint(0, tr_x.shape[0], (cfg.batch,))
        bx, by = tr_x[idx].to(dev), tr_y[idx].to(dev)
        with half_ctx:
            h = model.get_hidden(bx)
        tot_s, dH, dU, dV = head_loss_fn(h.reshape(-1, cfg.d), model.emb.U, model.emb.V,
                                         model.emb.R, by, model.emb.alpha, model.emb.rank,
                                         lc_buf, cfg.sub_batch, half=cfg.fp16, nce_k=nce_k,
                                         chunk=cfg.chunk)
        loss = tot_s / (cfg.batch * cfg.seq)
        with half_ctx:
            h.reshape(-1, cfg.d).backward(dH)
        with torch.no_grad():
            if model.emb.U.grad is None:
                model.emb.U.grad = dU
            else:
                model.emb.U.grad.add_(dU)
            if model.emb.V.grad is None:
                model.emb.V.grad = dV
            else:
                model.emb.V.grad.add_(dV)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); opt.zero_grad(set_to_none=True)

        if st % cfg.log_every == 0:
            el = time.time() - t0
            log(f'  st={st:5d} [{cur_head}] | '
                f'loss={loss:.3f} | tok/s={(st + 1) * cfg.batch * cfg.seq / el:.0f} | {el:.0f}s')
        if cfg.eval_every > 0 and st > 0 and st % cfg.eval_every == 0:
            ppl = evaluate(); best = min(best, ppl)
            log(f'  >>> EVAL st={st:5d} | val_ppl={ppl:.2f} | best={best:.2f}')
            if save_best:
                sd = model.state_dict_compact() if cfg.compact_ckpt else model.state_dict()
                torch.save({'step': st, 'model_state': sd, 'best_ppl': best,
                            'config': cfg.as_dict(), 'compact': cfg.compact_ckpt,
                            'r_seed': cfg.r_seed},
                           os.path.join(cfg.out_dir, f'best_{cfg.head}.pt'))

    ppl = evaluate() if cfg.eval_at_end else float('nan')
    best = min(best, ppl)
    test_ppl = float('nan')
    if cfg.test_eval_at_end and cfg.test_data and os.path.exists(cfg.test_data):
        with open(cfg.test_data, 'rb') as f:
            td = pickle.load(f)
        ts_x = torch.tensor(td.get('te_x', td.get('test_x')), dtype=torch.int32)
        ts_y = torch.tensor(td.get('te_y', td.get('test_y')), dtype=torch.int32)
        test_ppl = eval_ppl(model, ts_x, ts_y, chunk=cfg.chunk)
        log(f'  TEST | test_ppl={test_ppl:.2f} (held-out, not tuned on)')
    dur = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    res = {'name': name, 'ppl': ppl, 'best': best, 'params': pc, 'sec': dur,
           'vram': vram, 'tok_per_sec': cfg.steps * cfg.batch * cfg.seq / dur,
           'test_ppl': test_ppl}
    log(f'  DONE {name}: final_ppl={ppl:.2f} best={best:.2f} | {dur:.0f}s | peak {vram:.2f}GB')
    return res


def train_many(cfgs, log=_flog):
    results = []
    for cfg in cfgs:
        results.append(train(cfg, log=log))
    log('\n=== FINAL ===')
    for r in results:
        log(f'{r["name"]:40} {r["params"]:>9,} params | best_ppl={r["best"]:.2f} '
            f'final={r["ppl"]:.2f} | {r["sec"]:.0f}s | {r["vram"]:.2f}GB')
    return results
