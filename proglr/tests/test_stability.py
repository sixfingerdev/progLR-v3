"""proglr stability test suite.

Covers: analytic-vs-autograd gradchecks (exact + NCE), seed determinism,
multi-seed stability, config matrix (fp16/checkpoint/sub-batch/hybrid),
head-schedule switches, checkpoint round-trip, error handling, API smoke.

Run:  python proglr/tests/test_stability.py
"""
import os
import sys
import math
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn.functional as F

print = functools.partial(print, flush=True)  # live logs even when piped

from proglr import (TrainConfig, train, train_many, load_data, make_model,
                    ProgLR, head_loss_fn, eval_ppl)

LAW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(LAW, 'condenser', 'wt103_data.pkl')
OUT = os.path.join(os.environ.get('TEMP', LAW), 'opencode', 'proglr_stability')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda'
_TOL = 1e-3

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f'  PASS  {name}')
    except Exception as e:
        FAIL.append((name, e))
        print(f'  FAIL  {name}: {type(e).__name__}: {e}')


# ---------------------------------------------------------------- gradchecks
def gradcheck_common(y, R, U, V, h, alpha, rank, neg=None, chunk=None):
    """Analytic dH/dU/dV vs autograd for the same forward graph.

    head_loss_fn uses sum-reduction CE (same scale as train.py), so the
    reference must use reduction='sum' too.
    """
    s = alpha / rank
    lt = torch.einsum('nd,nd->n', h, R[y]) + s * torch.einsum('nd,nd->n', h @ V, U[y])
    if neg is not None:
        lc = torch.cat([lt.unsqueeze(1), h @ R[neg].t() + s * ((h @ V) @ U[neg].t())], dim=1)
        yc = torch.zeros(h.shape[0], dtype=torch.long, device=DEV)
    else:
        lc = h @ R.t() + s * ((h @ V) @ U.t())
        yc = y
    loss = F.cross_entropy(lc, yc, reduction='sum')
    gaH, gaU, gaV = torch.autograd.grad(loss, (h, U, V))

    n = h.shape[0]
    lc_buf = torch.empty(n, chunk if chunk else R.shape[0], device=DEV)
    tot, aH, aU, aV = head_loss_fn(h.detach(), U, V, R, y, alpha, rank, lc_buf, n,
                                   nce_k=(neg.numel() if neg is not None else 0),
                                   chunk=chunk if chunk else R.shape[0], neg=neg)
    assert abs(tot - loss.item()) < _TOL, f'loss {tot} vs {loss.item()}'
    for nm, a, g in [('dH', aH, gaH), ('dU', aU, gaU), ('dV', aV, gaV)]:
        err = (a - g).abs().max().item()
        assert err < _TOL, f'{nm} err {err:.2e}'
    return True


def test_nce_gradcheck():
    torch.manual_seed(0)
    h = torch.randn(16, 64, device=DEV, requires_grad=True)
    U = torch.randn(512, 8, device=DEV, requires_grad=True)
    V = torch.randn(64, 8, device=DEV, requires_grad=True)
    R = torch.randn(512, 64, device=DEV)
    y = torch.randint(0, 512, (16,), device=DEV)
    neg = torch.randint(0, 512, (32,), device=DEV)
    assert gradcheck_common(y, R, U, V, h, 64.0, 8, neg=neg)


def test_exact_gradcheck():
    torch.manual_seed(1)
    h = torch.randn(16, 64, device=DEV, requires_grad=True)
    U = torch.randn(512, 8, device=DEV, requires_grad=True)
    V = torch.randn(64, 8, device=DEV, requires_grad=True)
    R = torch.randn(512, 64, device=DEV)
    y = torch.randint(0, 512, (16,), device=DEV)
    assert gradcheck_common(y, R, U, V, h, 64.0, 8, neg=None)


def test_half_path_gradcheck():
    """fp16 path must still match autograd on the fp32 loss."""
    torch.manual_seed(2)
    h = torch.randn(16, 64, device=DEV, requires_grad=True)
    U = torch.randn(512, 8, device=DEV, requires_grad=True)
    V = torch.randn(64, 8, device=DEV, requires_grad=True)
    R = torch.randn(512, 64, device=DEV)
    y = torch.randint(0, 512, (16,), device=DEV)
    s = 64.0 / 8
    lc = h @ R.t() + s * ((h @ V) @ U.t())
    loss = F.cross_entropy(lc, y, reduction='sum')
    gaH, gaU, gaV = torch.autograd.grad(loss, (h, U, V))
    lc_buf = torch.empty(16, 512, device=DEV, dtype=torch.float16)
    tot, aH, aU, aV = head_loss_fn(h.detach(), U, V, R.half(), y, 64.0, 8, lc_buf, 16,
                                   half=True, nce_k=0, chunk=512)
    assert abs(tot - loss.item()) / loss.item() < 1e-3, \
        f'fp16 loss rel diff {(tot - loss.item()) / loss.item():.2e}'
    for nm, a, g in [('dH', aH, gaH), ('dU', aU, gaU), ('dV', aV, gaV)]:
        err = (a - g).abs().max().item()
        scale = max(a.abs().max().item(), g.abs().max().item())
        assert err < 2e-2 * scale, f'{nm} fp16-path rel err {err / scale:.2e}'


# ---------------------------------------------------------------- validation
def test_config_validation():
    def expect(exc, data=DATA, **kw):
        try:
            train(TrainConfig(data=data, out_dir=OUT, steps=1, **kw))
        except exc:
            return
        raise AssertionError(f'expected {exc.__name__} for {kw}')
    expect(ValueError, head='bogus')
    expect(FileNotFoundError, data='C:\\nope_missing.pkl')
    expect(ValueError, head_schedule=[(0, 'exact', 0), (-5, 'nce', 8)])
    expect(ValueError, head_schedule=[(0, 'exact', 0), (10, 'bogus', 8)])
    expect(ValueError, head_schedule=[(10, 'exact', 0), (5, 'nce', 8)])
    expect(ValueError, head_schedule=[(0, 'exact', 0), (999999, 'nce', 8)])
    expect(ValueError, batch=0)


# ---------------------------------------------------------------- training runs
def run(cfg, steps_override=None):
    if steps_override:
        cfg.steps = steps_override
    if not cfg.out_dir:
        cfg.out_dir = OUT
    if cfg.eval_every == 2000:   # untouched default -> keep tests fast
        cfg.eval_every = 0
    if cfg.save_best and cfg.eval_every == 0:  # no eval -> no checkpoint writes
        cfg.save_best = False
    return train(cfg, log=lambda *a, **k: None)


def test_determinism():
    a = run(TrainConfig(data=DATA, d=768, h=12, n_std=1, n_lr=1, steps=20,
                        log_every=10, seed=7), steps_override=None)
    b = run(TrainConfig(data=DATA, d=768, h=12, n_std=1, n_lr=1, steps=20,
                        log_every=10, seed=7), steps_override=None)
    rel = abs(a['ppl'] - b['ppl']) / max(a['ppl'], 1e-9)
    assert rel < 1e-3, f'rel ppl drift {rel:.2e} ({a["ppl"]} vs {b["ppl"]})'
    assert abs(a['sec'] / max(b['sec'], 1e-9) - 1) < 0.5


def test_multiseed():
    for seed in (1, 2, 3):
        r = run(TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=15,
                            log_every=15, seed=seed, head='nce', nce_k=128))
        assert not math.isnan(r['ppl']) and math.isfinite(r['ppl']), f'seed {seed} NaN/Inf ppl'


def test_config_matrix():
    cfgs = [
        TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=8, seed=7, head='exact'),
        TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=8, seed=7, head='exact', fp16=True),
        TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=8, seed=7, head='nce', nce_k=256),
        TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=8, seed=7, head='nce', nce_k=256, fp16=True),
        TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=8, seed=7, head='exact', checkpoint_all=True),
        TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=8, seed=7, head='nce', sub_batch=256),
        TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=8, seed=7, batch=4, head='nce'),
        TrainConfig(data=DATA, d=384, h=6, n_std=2, n_lr=2, steps=8, seed=7, head='nce', nce_k=64),
    ]
    for cfg in cfgs:
        r = run(cfg)
        assert math.isfinite(r['ppl']), f'non-finite ppl for {cfg.as_dict()}'
        assert r['vram'] < 6.0, f'VRAM {r["vram"]}GB over budget'


def test_schedule_multi_switch():
    r = run(TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=30, seed=7,
                        head_schedule=[(0, 'nce', 64), (10, 'exact', 0), (20, 'nce', 128)]))
    assert math.isfinite(r['ppl'])


def test_checkpoint_roundtrip():
    out = os.path.join(OUT, 'roundtrip')
    os.makedirs(out, exist_ok=True)
    r = run(TrainConfig(data=DATA, d=384, h=6, n_std=1, n_lr=1, steps=30, seed=7,
                        head='exact', save_best=True, eval_every=15, out_dir=out))
    ck = torch.load(os.path.join(out, 'best_exact.pt'), weights_only=False)
    vs, tr_x, tr_y, te_x, te_y = load_data(DATA)
    m = make_model(vs, TrainConfig(d=384, h=6, n_std=1, n_lr=1))
    m.load_state_dict(ck['model_state'])
    m.to(DEV)
    ppl = eval_ppl(m, te_x, te_y)
    assert ppl >= ck['best_ppl'] * (1 - 1e-4), f'roundtrip ppl {ppl} < ckpt best {ck["best_ppl"]}'
    assert ck['config']['head'] == 'exact'


def test_api_smoke():
    vs, *_ = load_data(DATA)
    m = ProgLR(vs, 128, 4, 1, 1)
    m.to(DEV)
    xs = torch.randint(0, vs, (2, 32), device=DEV)
    h = m.get_hidden(xs)
    assert h.shape == (2, 32, 128)
    nce = TrainConfig(data=DATA, out_dir=OUT, d=384, h=6, n_std=1, n_lr=1,
                      steps=5, seed=7, head='nce', nce_k=32, eval_at_end=True)
    exact = TrainConfig(data=DATA, out_dir=OUT, d=384, h=6, n_std=1, n_lr=1,
                        steps=5, seed=7, head='exact', eval_at_end=True)
    res = train_many([nce, exact], log=lambda *a, **k: None)
    assert len(res) == 2 and all(math.isfinite(r.get('ppl', 1.0)) for r in res)


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('highest')
    print(f'proglr stability suite | data={DATA} | out={OUT}')
    for nm, fn in [('nce_gradcheck', test_nce_gradcheck),
                   ('exact_gradcheck', test_exact_gradcheck),
                   ('half_path_gradcheck', test_half_path_gradcheck),
                   ('config_validation', test_config_validation),
                   ('determinism', test_determinism),
                   ('multiseed', test_multiseed),
                   ('config_matrix', test_config_matrix),
                   ('schedule_multi_switch', test_schedule_multi_switch),
                   ('checkpoint_roundtrip', test_checkpoint_roundtrip),
                   ('api_smoke', test_api_smoke)]:
        check(nm, fn)
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for nm, e in FAIL:
        print(f'  FAILED: {nm} -> {type(e).__name__}: {e}')
    sys.exit(1 if FAIL else 0)
