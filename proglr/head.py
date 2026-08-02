"""Analytic head: exact chunked softmax and NCE (sampled) paths + eval.

Exact path: l_ic = h_i R_c^T + s (h_i V) U_c^T computed chunk-wise on a reused buffer,
in-place softmax, analytic gradients dH/dU/dV (no autograd through the 250K-class logits).
NCE path: target logits + K shared random negatives; logits/softmax in fp32 (TF32) to
avoid fp16 overflow of large target logits (inf -> NaN).
"""
import math
import torch
import torch.nn.functional as F


def head_loss_fn(h, U, V, R, y, alpha, rank, lc_buf, sub_batch, half=False, nce_k=0,
                 chunk=50000, neg=None):
    """Loss (detached scalar) + analytic dH, dU, dV for exact CE or NCE.

    Args:
        h: (N, D) hidden states (grad-requires)
        U: (V, rank) embedding/head factor (second of the weight pair)
        V: (D, rank) embedding/head factor (first of the weight pair)
        R: (V, D) frozen base embedding (half dtype when fp16 mode)
        y: (N,) target ids
        lc_buf: reusable (sub_batch, chunk) logit buffer
        half: cast R/U/V per call to fp16 tensor cores (grads accumulate fp32)
        nce_k: >0 enables the NCE path with k shared negatives per step
        chunk: vocab chunk width for the exact path
        neg: optional fixed negative ids (K,) — injectable for reproducibility/gradcheck;
             default draws a fresh random shared set per call
    """
    s = alpha / rank
    N, D = h.shape
    dev = h.device
    dH = torch.zeros_like(h); dU = torch.zeros_like(U); dV = torch.zeros_like(V)
    hf = h.half() if half else h
    tf = hf @ (V.half() if half else V)
    total = 0.0
    y = y.reshape(-1)

    if nce_k > 0:
        if neg is None:
            with torch.no_grad():
                neg = torch.randint(0, R.shape[0], (nce_k,), device=dev)  # shared negatives (K,)
        Rg = R[neg].float(); Ug = U[neg].float(); Ry = R[y].float(); Uy = U[y].float()
        hf32 = hf.float(); tf32 = tf.float()
        Vf = V.float() if half else V
        lt = torch.einsum('nd,nd->n', hf32, Ry) + s * torch.einsum('nd,nd->n', tf32, Uy)
        lc = torch.cat([lt.unsqueeze(1), hf32 @ Rg.t() + s * (tf32 @ Ug.t())], dim=1)
        total += F.cross_entropy(lc, torch.zeros(N, dtype=torch.long, device=dev),
                                 reduction='sum').item()
        torch.softmax(lc, dim=-1, out=lc)
        g = lc
        g[:, 0] -= 1.0
        dH.add_(g[:, 1:] @ Rg)
        dH.add_(s * (g[:, 1:] @ Ug) @ Vf.t())
        dH.add_(g[:, 0:1] * Ry)
        dH.add_(s * (g[:, 0:1] * Uy) @ Vf.t())
        dU.index_add_(0, y, s * g[:, 0:1] * tf32)
        dU.index_add_(0, neg, s * g[:, 1:].t() @ tf32)
        dV.add_(s * (hf32.t() @ (g[:, 1:] @ Ug + g[:, 0:1] * Uy)))
        return total, dH, dU, dV

    nch = math.ceil(R.shape[0] / chunk)
    with torch.no_grad():
        for i0 in range(0, N, sub_batch):
            i1 = min(N, i0 + sub_batch)
            n = i1 - i0
            h_b = hf[i0:i1]; t_b = tf[i0:i1]; y_b = y[i0:i1]
            lc = lc_buf[:n]
            for c in range(nch):
                Rc = R[c*chunk:(c+1)*chunk]
                Uc = U[c*chunk:(c+1)*chunk]
                if half:
                    Rc = Rc.half(); Uc = Uc.half()
                torch.addmm(lc, h_b, Rc.t(), beta=0, alpha=1, out=lc)
                torch.addmm(lc, t_b, Uc.t(), beta=1, alpha=s, out=lc)
                yc = torch.where((y_b >= c*chunk) & (y_b < (c+1)*chunk), y_b - c*chunk, -100)
                valid = yc >= 0
                nv = valid.sum().item()
                if nv == 0:
                    continue
                total += F.cross_entropy(lc.float(), yc.long(), reduction='sum').item()
                torch.softmax(lc, dim=-1, out=lc)
                g = lc
                tg = yc.clamp_min(0).long()
                g.scatter_add_(1, tg.unsqueeze(1), torch.full_like(g, -1.0))
                g[~valid] = 0.0
                dU[c*chunk:(c+1)*chunk].add_((s * (g.t() @ t_b)).float())
                dH[i0:i1].add_((g @ Rc).float())
                dH[i0:i1].add_(((s * (g @ Uc)).float() @ V.t()))
                dV.add_((s * (h_b.t() @ (g @ Uc))).float())
    return total, dH, dU, dV


def eval_ppl(model, xs, ys, chunk=50000, dim=None):
    """Full-softmax val PPL (always exact CE, regardless of training head)."""
    dev = next(model.parameters()).device
    model.eval(); tot_s = 0.0; tot_v = 0
    if dim is None:
        dim = model.pos.shape[-1]
    half = model.emb.R.dtype == torch.float16
    vs = model.emb.R.shape[0]
    lc = torch.empty(1024, chunk, device=dev, dtype=torch.float16 if half else torch.float32)
    with torch.no_grad():
        for i in range(0, xs.shape[0], 8):
            bx = xs[i:i + 8].to(dev); by = ys[i:i + 8].to(dev)
            h = model.get_hidden(bx).reshape(-1, dim)
            hf = h.half() if half else h
            s = model.emb.alpha / model.emb.rank
            t = hf @ (model.emb.V.half() if half else model.emb.V)
            y = by.reshape(-1)
            n = hf.shape[0]
            for i0 in range(0, n, 1024):
                i1 = min(n, i0 + 1024); nb = i1 - i0
                lc_n = lc[:nb]
                for c in range(math.ceil(vs / chunk)):
                    Rc = model.emb.R[c*chunk:(c+1)*chunk]; Uc = model.emb.U[c*chunk:(c+1)*chunk]
                    if half:
                        Rc = Rc.half(); Uc = Uc.half()
                    torch.addmm(lc_n, hf[i0:i1], Rc.t(), beta=0, alpha=1, out=lc_n)
                    torch.addmm(lc_n, t[i0:i1], Uc.t(), beta=1, alpha=s, out=lc_n)
                    yc = torch.where((y[i0:i1] >= c*chunk) & (y[i0:i1] < (c+1)*chunk),
                                     y[i0:i1] - c*chunk, -100)
                    valid = yc >= 0
                    nv = valid.sum().item()
                    if nv > 0:
                        tot_s += F.cross_entropy(lc_n.float(), yc.long(), reduction='sum').item()
                        tot_v += nv
    return math.exp(tot_s / max(tot_v, 1))
