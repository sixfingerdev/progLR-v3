"""World-knowledge test on ckpt_12000.pt. Self-contained model classes (copied, no training import)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt

torch.set_float32_matmul_precision('high')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from condenser.v3_embed import V3WeightEmbedding

D, H, R, A = 1024, 16, 16, 64
PATTERN = ['std', 'std'] + ['lr'] * 4 + ['std'] + ['lr'] * 4 + ['std']
VS = 50000

class LRLayer(nn.Module):
    def __init__(self, dim, heads, r=16, a=64):
        super().__init__()
        self.h, self.dh, self.r, self.a = heads, dim // heads, r, a
        for nm, od, id_ in [('q', dim, dim), ('k', dim, dim), ('v', dim, dim), ('o', dim, dim), ('ff1', 4 * dim, dim), ('ff2', dim, 4 * dim)]:
            setattr(self, f'U_{nm}', nn.Parameter(torch.randn(od, r) * 0.02))
            setattr(self, f'V_{nm}', nn.Parameter(torch.randn(id_, r) * 0.02))
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    def _l(self, x, nm):
        s = self.a / self.r
        return s * ((x @ getattr(self, f'V_{nm}')) @ getattr(self, f'U_{nm}').t())
    def forward(self, x):
        B, T, D = x.shape; h = self.dh; n = self.n1(x)
        Q = self._l(n, 'q').view(B, T, self.h, h).transpose(1, 2)
        K = self._l(n, 'k').view(B, T, self.h, h).transpose(1, 2)
        V_ = self._l(n, 'v').view(B, T, self.h, h).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V_, is_causal=True)
        x = x + self._l(a.transpose(1, 2).reshape(B, T, D), 'o')
        n2 = self.n2(x)
        return x + self._l(F.relu(self._l(n2, 'ff1')), 'ff2')

class StdLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.dh = dim // heads
        self.W_Q = nn.Linear(dim, dim, bias=False); self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False); self.W_O = nn.Linear(dim, dim, bias=False)
        self.W1 = nn.Linear(dim, dim * 4, bias=False); self.W2 = nn.Linear(dim * 4, dim, bias=False)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
    def forward(self, x):
        B, T, D = x.shape; h = self.dh
        n = self.n1(x)
        Q = self.W_Q(n).view(B, T, self.h, h).transpose(1, 2)
        K = self.W_K(n).view(B, T, self.h, h).transpose(1, 2)
        V = self.W_V(n).view(B, T, self.h, h).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        x = x + self.W_O(a.transpose(1, 2).reshape(B, T, D))
        n = self.n2(x)
        return x + self.W2(F.relu(self.W1(n)))

class Model(nn.Module):
    def __init__(self, vs, D, H, pattern, r=16, a=64):
        super().__init__()
        self.emb = V3WeightEmbedding(vs, D, r, a)
        self.pos = nn.Parameter(torch.randn(1, 1024, D) * 0.01)
        self.layers = nn.ModuleList(
            StdLayer(D, H) if k == 'std' else LRLayer(D, H, r, a) for k in pattern)
        self.norm = nn.LayerNorm(D)
    def get_hidden(self, x):
        B, T = x.shape
        h = self.emb.forward_embed(x) + self.pos[:, :T]
        for l in self.layers:
            h = ckpt.checkpoint(l, h, use_reentrant=False)
        return self.norm(h)

dev = torch.device('cuda')

print('Loading ckpt_12000.pt...', flush=True)
ck = torch.load(os.path.join(BASE, 'report_out_sandwich_bpe', 'ckpt_12000.pt'),
                map_location=dev, weights_only=False)
print(f"step={ck['step']} best_ppl={ck['best_ppl']:.2f}", flush=True)

model = Model(VS, D, H, PATTERN, R, A).to(dev)
model.load_state_dict(ck['model_state'])
model.eval()

from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(BASE, 'condenser', 'wt103_bpe_tokenizer.json'))

def sample(logits, temp=0.7, top_k=40, top_p=0.9, rep=1.15, used=None):
    logits = logits / temp
    if used:
        for t in used:
            logits[t] = logits[t] / rep if logits[t] > 0 else logits[t] * rep
    if top_k:
        k = min(top_k, logits.numel())
        logits = logits.masked_fill(logits < torch.topk(logits, k).values[-1], float('-inf'))
    if top_p < 1.0:
        s, i = torch.sort(logits, descending=True)
        c = torch.cumsum(F.softmax(s, -1), -1)
        s[c - F.softmax(s, -1) > top_p] = float('-inf')
        logits = torch.gather(s, -1, i.argsort(-1))
    return torch.multinomial(F.softmax(logits, -1), 1).item()

def gen(prompt, steps=22):
    ids = tok.encode(prompt.lower()).ids
    ctx = torch.tensor([ids], dtype=torch.long, device=dev)
    out = list(ids); used = ids[:]
    with torch.no_grad():
        for _ in range(steps):
            h = model.get_hidden(ctx[:, -512:])
            x = h[:, -1, :]
            logits = (x @ model.emb.R.t()) + (model.emb.alpha / model.emb.rank) * ((x @ model.emb.V) @ model.emb.U.t())
            nxt = sample(logits[0], used=used[-128:])
            out.append(nxt); used.append(nxt)
            ctx = torch.cat([ctx, torch.tensor([[nxt]], device=dev)], dim=1)
    return tok.decode(out)

prompts = [
    'the capital of france is',
    'the united states declared independence in',
    'einstein developed the theory of',
    'water boils at',
    'the second world war ended in',
    'new york is a city in the state of',
    'the earth orbits around',
    'shakespeare wrote the play',
    'the largest planet in the solar system is',
    'the first president of the united states was',
]
for p in prompts:
    print(f'Q: {p}', flush=True)
    print(f'A: {gen(p)}', flush=True)
    print(flush=True)
