"""
Finetune: expand head to 250K vocab (transfer old rows, new rows random).
Layers + R + V + old U rows FROZEN; only new U rows (50000:) are trained.
All head gradients ANALYTIC inside torch.no_grad() â€” no autograd backward
through layers, no graphs. Chunked CE (5x50K) to fit 6GB. Optimizer touches
ONLY the new-row slice U[NEW_START:] (full-tensor Adam moved old rows ~1e-5/step
even with zero grad -> old-vocab CE creep; see diag runs).
Weighted per-position loss (empty chunks skipped -> no NaN).
"""
import os, sys, math, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'condenser', 'wt103_data_250k.pkl')
VOCAB = os.path.join(BASE, 'condenser', 'wt103_vocab_250k.pkl')
CKPT_IN = os.path.join(BASE, 'report_out', 'checkpoints', 'ckpt_16000.pt')
OUT_DIR = os.path.join(BASE, 'report_out_250k')
CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

# === Low-Rank Layer (identical to trained model) ===
class LRLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, U, V, a, r):
        s=a/r; W=s*(U@V.t())
        ctx.save_for_backward(x,U,V); ctx.s=s
        return F.linear(x, W)
    @staticmethod
    def backward(ctx, go):
        x,U,V=ctx.saved_tensors
        i_f=x.shape[-1]; o_f=go.shape[-1]
        x_f=x.reshape(-1,i_f); go_f=go.reshape(-1,o_f)
        gW=go_f.t()@x_f; s=ctx.s
        return (F.linear(go,s*(V@U.t())), s*(gW@V), s*(gW.t()@U), None, None)

class LRLayer(nn.Module):
    def __init__(self, dim, heads, r=16, a=64):
        super().__init__()
        self.h,self.dh,self.r,self.a=heads,dim//heads,r,a
        for nm,od,id_ in [('q',dim,dim),('k',dim,dim),('v',dim,dim),('o',dim,dim),('ff1',4*dim,dim),('ff2',dim,4*dim)]:
            setattr(self,f'U_{nm}',nn.Parameter(torch.randn(od,r)*0.02))
            setattr(self,f'V_{nm}',nn.Parameter(torch.randn(id_,r)*0.02))
        self.n1,self.n2=nn.LayerNorm(dim),nn.LayerNorm(dim)
    def _l(self,x,nm):
        return LRLinearFn.apply(x,getattr(self,f'U_{nm}'),getattr(self,f'V_{nm}'),self.a,self.r)
    def forward(self,x):
        B,T,D=x.shape; h=self.dh; n=self.n1(x)
        Q=self._l(n,'q').view(B,T,self.h,h).transpose(1,2)
        K=self._l(n,'k').view(B,T,self.h,h).transpose(1,2)
        V_=self._l(n,'v').view(B,T,self.h,h).transpose(1,2)
        a=F.scaled_dot_product_attention(Q,K,V_,is_causal=True)
        o=a.transpose(1,2).reshape(B,T,D)
        x=x+self._l(o,'o'); n2=self.n2(x)
        x=x+self._l(F.relu(self._l(n2,'ff1')),'ff2')
        return x

class StdLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h=heads; self.dh=dim//heads
        self.W_Q=nn.Linear(dim,dim,bias=False); self.W_K=nn.Linear(dim,dim,bias=False)
        self.W_V=nn.Linear(dim,dim,bias=False); self.W_O=nn.Linear(dim,dim,bias=False)
        self.W1=nn.Linear(dim,dim*4,bias=False); self.W2=nn.Linear(dim*4,dim,bias=False)
        self.n1=nn.LayerNorm(dim); self.n2=nn.LayerNorm(dim)
    def forward(self,x):
        B,T,D=x.shape; h=self.h; dh=self.dh; n=self.n1(x)
        Q=self.W_Q(n).view(B,T,h,dh).transpose(1,2)
        K=self.W_K(n).view(B,T,h,dh).transpose(1,2)
        V=self.W_V(n).view(B,T,h,dh).transpose(1,2)
        a=F.scaled_dot_product_attention(Q,K,V,is_causal=True)
        x=x+self.W_O(a.transpose(1,2).reshape(B,T,D))
        n=self.n2(x)
        return x+self.W2(F.relu(self.W1(n)))

# === V3-Embed with chunked head ===
class V3WeightEmbeddingX(nn.Module):
    def __init__(self, vocab_size, hidden_dim, rank=8, alpha=1.0):
        super().__init__()
        self.rank = rank; self.alpha = alpha
        R = torch.randn(vocab_size, hidden_dim) / math.sqrt(hidden_dim)
        self.register_buffer('R', R)
        self.U = nn.Parameter(torch.randn(vocab_size, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)

    def forward_embed(self, input_ids):
        R_emb = F.embedding(input_ids, self.R)
        U_emb = F.embedding(input_ids, self.U)
        lora = (self.alpha / self.rank) * (U_emb @ self.V.T)
        return R_emb + lora

    def forward_head_chunk(self, x, c, chunk):
        s = self.alpha / self.rank
        Rc = self.R[c*chunk:(c+1)*chunk]
        Uc = self.U[c*chunk:(c+1)*chunk]
        return F.linear(x, Rc) + s * ((x @ self.V) @ Uc.t())

class ProgLR250(nn.Module):
    def __init__(self, vs, D, L, H, num_std=1, r=16, a=64):
        super().__init__()
        self.emb = V3WeightEmbeddingX(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1, 1024, D) * 0.01)
        self.layers = nn.ModuleList()
        for i in range(L):
            if i < num_std:
                self.layers.append(StdLayer(D, H))
            else:
                self.layers.append(LRLayer(D, H, r, a))
        self.norm = nn.LayerNorm(D)
    def get_hidden(self, x):
        B,T=x.shape
        h=self.emb.forward_embed(x)+self.pos[:,:T]
        for l in self.layers: h=l(h)
        return self.norm(h)

# === Load 250K data ===
print('Loading data...', flush=True)
with open(DATA, 'rb') as f:
    d = pickle.load(f)
VS = d['vocab_size']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tr_x = torch.tensor(d['tr_x'], device=dev, dtype=torch.int32)
tr_y = torch.tensor(d['tr_y'], device=dev, dtype=torch.int32)
te_x = torch.tensor(d['te_x'], device=dev, dtype=torch.int32)
te_y = torch.tensor(d['te_y'], device=dev, dtype=torch.int32)
del d

with open(VOCAB, 'rb') as f:
    voc = pickle.load(f)
id2word = voc['id2word']
v2i = {v: k for k, v in id2word.items()}

# === Config ===
B=8; D=768; H=12; R=16; A=64; SEQ=512
OLD_VS=50000; NEW_START=OLD_VS; CHUNK=50000; NCH=math.ceil(VS/CHUNK)
S=4000
print(f'Vocab: {VS}, Train: {tr_x.shape[0]:,}, Valid: {te_x.shape[0]:,}, Chunks: {NCH}', flush=True)

# New-word density per train chunk -> oversample new-word-rich chunks
rho = (tr_y >= NEW_START).float().mean(dim=1)  # (198097,)
TOPK_N = max(tr_x.shape[0] // 100, 1)  # top 1% densest chunks
topk_idx = rho.topk(TOPK_N).indices.to('cpu')
print(f'New-word density: mean={rho.mean().item():.4f} max={rho.max().item():.4f} | '
      f'top-1% mean={rho[topk_idx].mean().item():.4f} (x{rho[topk_idx].mean().item()/rho.mean().item():.0f})', flush=True)

def sample_chunks(B):
    """50% from top-1% dense chunks, 50% uniform (all on CPU -> fast multinomial)."""
    n = tr_x.shape[0]
    if torch.rand(1).item() < 0.5:
        return topk_idx[torch.randint(0, TOPK_N, (B,))]
    return torch.randint(0, n, (B,))

# === Build expanded model & transfer weights ===
model = ProgLR250(VS, D, 2, H, 1, R, A).to(dev)
ckpt = torch.load(CKPT_IN, map_location=dev)
sd = ckpt['model_state']
with torch.no_grad():
    model.emb.R[:OLD_VS] = sd['emb.R']
    model.emb.U[:OLD_VS] = sd['emb.U']
    model.emb.V.copy_(sd['emb.V'])
    model.pos.copy_(sd['pos'])
    model.norm.load_state_dict({'weight': sd['norm.weight'], 'bias': sd['norm.bias']})
    for li in range(2):
        sub = {k.replace(f'layers.{li}.', ''): v for k, v in sd.items() if k.startswith(f'layers.{li}.')}
        model.layers[li].load_state_dict(sub)
# Freeze everything except new U rows
for p in model.parameters():
    p.requires_grad_(False)
model.emb.U.requires_grad_(True)
n_train = model.emb.U.shape[0] - NEW_START
print(f'Params: {sum(p.numel() for p in model.parameters()):,} | trainable: {n_train*model.emb.U.shape[1]:,} (new U rows only)', flush=True)

# === Loss: weighted per-position CE over chunks (skips empty chunks -> no NaN) ===
# Speed: ONE big buffer (lc_buf) reused across chunks; logits computed as a
# single fused matmul addmm(out=) of [hf, s*t] @ [Rc, Uc].t() (cat trick);
# softmax done in-place on lc_buf. No 819MB temporaries -> no cudaMalloc storms.
def _logits_step(out, hf_cat, Rc, Uc):
    Rc_cat = torch.cat([Rc, Uc], dim=1)  # (CHUNK, D+r), ~157MB
    torch.addmm(out, hf_cat, Rc_cat.t(), beta=0, alpha=1, out=out)

def chunk_loss(model, x, y, h, lc_buf, return_dU=True):
    """Returns (sum_loss, n_valid, dU, sum_new, n_new). dU is the gradient for
    the NEW-row slice only (shape [200000, r]); old rows are never touched.
    Chunk dU scaled by (1 + 9*rho_c) where rho_c = new-word fraction in chunk
    (focuses gradient on rare/new words). lc_buf: preallocated (B*T, CHUNK)."""
    D = h.shape[-1]
    s = model.emb.alpha / model.emb.rank
    r = model.emb.rank
    tot_s = 0.0; tot_v = 0
    tot_s_new = 0.0; tot_v_new = 0
    dU = torch.zeros_like(model.emb.U[NEW_START:]) if return_dU else None
    h_f = h.detach().reshape(-1, D)
    t = h_f @ model.emb.V.detach()  # (B*T, r), shared across chunks
    hf_cat = torch.cat([h_f, s * t], dim=1)  # (B*T, D+r), tiny
    y = y.reshape(-1)
    for c in range(NCH):
        Rc = model.emb.R[c*CHUNK:(c+1)*CHUNK]
        Uc = model.emb.U[c*CHUNK:(c+1)*CHUNK].detach()
        _logits_step(lc_buf, hf_cat, Rc, Uc)
        yc = torch.where((y >= c*CHUNK) & (y < (c+1)*CHUNK), y - c*CHUNK, -100)
        valid = yc >= 0
        nv = valid.sum()
        if nv > 0:
            tot_s += F.cross_entropy(lc_buf, yc.long(), reduction='sum').item()
            tot_v += nv.item()
            yc_new = torch.where((y >= NEW_START) & valid, yc, -100)
            nv_new = (yc_new >= 0).sum().item()
            if nv_new > 0:
                tot_s_new += F.cross_entropy(lc_buf, yc_new.long(), reduction='sum').item()
                tot_v_new += nv_new
            if return_dU and c*CHUNK >= NEW_START:  # chunks fully below NEW_START are frozen old rows
                mask = valid.unsqueeze(1)
                tg = yc.clamp_min(0).long()
                lc_buf.sub_(lc_buf.max(dim=-1, keepdim=True).values)  # in-place softmax
                lc_buf.exp_()  # SUM-CE gradient: no /nv (full-strength pull per occurrence)
                p_t_t = lc_buf.t() @ t  # (CHUNK, r)
                p_t_t.sub_(torch.zeros((CHUNK, r), device=t.device).scatter_add_(0, tg.unsqueeze(1).expand(-1, r), t * mask))
                p_t_t.sub_(p_t_t.mean(dim=0, keepdim=True))  # remove softmax-invariant row-mean (kills uniform Adam drift)
                rho_c = valid.float().mean().item()  # new-word fraction in this chunk
                dU[c*CHUNK-NEW_START:(c+1)*CHUNK-NEW_START].add_(s * (1 + 9.0 * rho_c) * p_t_t)
    return tot_s, tot_v, dU, tot_s_new, tot_v_new

def eval_ppl(model, lc_buf):
    model.eval(); tot_s=0.0; tot_v=0; old_s=0.0; old_v=0; new_s=0.0; new_v=0
    D = model.emb.R.shape[1]; r = model.emb.rank
    s = model.emb.alpha / model.emb.rank
    lc_eval = torch.empty(lc_buf.shape, device=lc_buf.device)  # dedicated, never resized
    with torch.no_grad():
        for i in range(0, te_x.shape[0], 8):
            bx=te_x[i:i+8]; by=te_y[i:i+8]
            h = model.get_hidden(bx)
            h_f = h.reshape(-1, h.shape[-1])
            t = h_f @ model.emb.V
            hf_cat = torch.cat([h_f, s * t], dim=1)
            y = by.reshape(-1)
            for c in range(NCH):
                Rc = model.emb.R[c*CHUNK:(c+1)*CHUNK]
                Uc = model.emb.U[c*CHUNK:(c+1)*CHUNK]
                _logits_step(lc_eval[:h_f.shape[0]], hf_cat, Rc, Uc)
                yc = torch.where((y >= c*CHUNK) & (y < (c+1)*CHUNK), y - c*CHUNK, -100)
                valid = yc >= 0
                nv = valid.sum().item()
                if nv > 0:
                    tot_s += F.cross_entropy(lc_eval[:h_f.shape[0]], yc.long(), reduction='sum').item()
                    tot_v += nv
                old = (y < NEW_START) & valid
                nv_old = old.sum().item()
                if nv_old > 0:
                    old_s += F.cross_entropy(lc_eval[:h_f.shape[0]], torch.where(old, yc, -100).long(), reduction='sum').item()
                    old_v += nv_old
                new = (y >= NEW_START) & valid
                nv_new = new.sum().item()
                if nv_new > 0:
                    new_s += F.cross_entropy(lc_eval[:h_f.shape[0]], torch.where(new, yc, -100).long(), reduction='sum').item()
                    new_v += nv_new
    return (math.exp(tot_s/tot_v), old_s/max(old_v,1), new_s/max(new_v,1))

def generate(model, prompt_ids, max_new=40, temp=0.8, top_k=50):
    model.eval()
    inp = torch.tensor([prompt_ids], device=dev, dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_new):
            h = model.get_hidden(inp[:, -SEQ:])
            last = h[:, -1:, :]
            logits = torch.cat([model.emb.forward_head_chunk(last, c, CHUNK) for c in range(NCH)], dim=-1)
            lg = logits[0, 0]
            lg[0] = -float('inf'); lg[1] = -float('inf')  # block <pad>, <unk>
            probs = F.softmax(lg / temp, dim=-1)
            topk = torch.topk(probs, top_k)
            nxt = topk.indices[torch.multinomial(topk.values, 1)].view(1,1)
            inp = torch.cat([inp, nxt], dim=1)
    return [id2word.get(t.item(), '?') for t in inp[0]]

if __name__ == '__main__':
    WARMUP=200
    def lr_at(st):
        if st < WARMUP:
            return PEAK_LR * (st+1)/WARMUP
        t = (st-WARMUP)/(S-WARMUP)
        return PEAK_LR * 0.5 * (1 + math.cos(math.pi * min(t,1.0)))

    print(f'\nStarting finetune: B={B}, steps={S}, only new U rows train (oversampled)\n', flush=True)
    t_start=time.time(); best_ppl=float('inf')
    lc_buf = torch.empty(B*SEQ, CHUNK, device=dev)
    PEAK_LR = 3e-3
    RESUME = os.environ.get('RESUME_CKPT', '')
    if RESUME:
        sd = torch.load(RESUME, map_location=dev)['model_state']
        with torch.no_grad():
            model.emb.U.copy_(sd['emb.U'])
        print(f'Resumed U from {RESUME}', flush=True)
    ppl0, old0, new0 = eval_ppl(model, lc_buf)
    print(f'  st=pre | val_ppl={ppl0:.2f} | old_ce={old0:.3f} | new_ce={new0:.2f}', flush=True)
    torch.cuda.empty_cache()
    # Manual Adam on the NEW-row slice only: old rows are structurally excluded
    # from every optimizer op (torch's AdamW moved old rows ~1e-5/step even
    # with zero grad -> old-vocab CE creep; a Parameter slice is non-leaf, so
    # a torch optimizer can't target the slice anyway).
    U_sl = model.emb.U[NEW_START:]
    m = torch.zeros_like(U_sl); v = torch.zeros_like(U_sl); tstep = 0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for st in range(S):
        lr = lr_at(st)
        idx = sample_chunks(B).to(dev)
        bx, by = tr_x[idx], tr_y[idx]
        with torch.no_grad():
            h = model.get_hidden(bx)
            tot_s, tot_v, dU, tot_s_new, tot_v_new = chunk_loss(model, bx, by, h, lc_buf)
        tstep += 1
        m.mul_(b1).add_(dU, alpha=1 - b1)
        v.mul_(b2).addcmul_(dU, dU, value=1 - b2)
        with torch.no_grad():
            U_sl.addcdiv_(m / (1 - b1 ** tstep), (v / (1 - b2 ** tstep)).sqrt().add_(eps), value=-lr)
        loss = tot_s / max(tot_v, 1)
        loss_new = tot_s_new / max(tot_v_new, 1)

        if st % 15 == 0:
            elapsed = time.time()-t_start
            tok_per_sec = (st+1)*B*SEQ/elapsed if elapsed > 0 else 0
            print(f'  st={st:>4} | lr={lr:.1e} | loss={loss:.3f} | new={loss_new:.2f} | tok/s={tok_per_sec:.0f} | elapsed={elapsed:.0f}s', flush=True)

        if st > 0 and st % 2000 == 0:
            ppl, old_ce, new_ce = eval_ppl(model, lc_buf); best_ppl=min(best_ppl,ppl)
            gen_words = generate(model, [v2i.get('the',2)], 40)
            gen_text = ' '.join(gen_words).replace(' @','').replace(' ,',',').replace(' .','.')
            print(f'  >>> EVAL st={st:>4} | val_ppl={ppl:.2f} | best={best_ppl:.2f} | old_ce={old_ce:.3f} | new_ce={new_ce:.2f}', flush=True)
            print(f'  >>> GEN: {gen_text}', flush=True)

        if st > 0 and st % 500 == 0:
            torch.save({'step': st, 'model_state': model.state_dict(),
                        'best_ppl': best_ppl},
                       os.path.join(CKPT_DIR, f'ckpt_{st:05d}.pt'))
            print(f'  >>> Saved ckpt_{st:05d}', flush=True)

    ppl, old_ce, new_ce = eval_ppl(model, lc_buf); best_ppl=min(best_ppl,ppl)
    gen_words = generate(model, [v2i.get('the',2)], 40)
    gen_text = ' '.join(gen_words).replace(' @','').replace(' ,',',').replace(' .','.')
    duration=time.time()-t_start
    print(f'Done! {duration:.0f}s | val_ppl={ppl:.2f} | best={best_ppl:.2f} | old_ce={old_ce:.3f} | new_ce={new_ce:.2f}', flush=True)
    print(f'GEN: {gen_text}', flush=True)
