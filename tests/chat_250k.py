"""
Interactive terminal chat with the final 250K model (report_out_250k/checkpoints/ckpt_03500.pt).
Streamed, token-by-token generation (greedy or sampled).
Each message is a fresh prompt (the model is a raw WT103 LM, not instruction-tuned).
Commands: /help /quit /len N /temp T /topk K /greedy
"""
import os, sys, re, math, pickle, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


CKPT = os.path.join(BASE, 'report_out_250k', 'checkpoints', 'ckpt_03500.pt')
VOCAB = os.path.join(BASE, 'condenser', 'wt103_vocab_250k.pkl')

# === Model (identical to finetune_vocab250k.py) ===
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

# === Load ===
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(VOCAB, 'rb') as f:
    voc = pickle.load(f)
id2word = voc['id2word']
v2i = voc['vocab']
CHUNK = 50000

ckpt = torch.load(CKPT, map_location=dev)
VS = ckpt['model_state']['emb.R'].shape[0]
NCH = math.ceil(VS / CHUNK)
D, H = 768, 12
model = ProgLR250(VS, D, 2, H, 1, 16, 64).to(dev)
model.load_state_dict(ckpt['model_state'])
model.eval()
step = ckpt.get('step', '?')
best_ppl = ckpt.get('best_ppl', '?')
print(f'Loaded {CKPT}')
print(f'  step={step} | best_ppl={best_ppl} | vocab={VS:,} | device={dev} | params={sum(p.numel() for p in model.parameters()):,}', flush=True)

UNK = v2i['<unk>']
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[,.;:!?()\"-]+")
NO_SPACE = set(',.;:!?()"')
MAX_CTX = 512

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

def encode(tokens):
    return [v2i.get(t, UNK) for t in tokens]

def decode_piece(tok):
    if '\ufffd' in tok: return ''
    if tok == '@-@': return '-'
    if tok == '@': return ''
    if tok.startswith("'"): return tok
    return tok if tok in NO_SPACE else ' ' + tok

def next_logits(model, ids, dev):
    x = torch.tensor([ids], device=dev, dtype=torch.int32)
    with torch.inference_mode():
        h = model.get_hidden(x)
        last = h[:, -1:, :]
        lgs = [model.emb.forward_head_chunk(last, c, CHUNK) for c in range(NCH)]
        lg = torch.cat(lgs, dim=-1)[0, 0]
        lg[0] = float('-inf'); lg[1] = float('-inf')  # block <pad>, <unk>
        return lg

def sample(logits, temp, top_k, top_p, rep_pen, ids):
    if temp <= 0:
        return int(logits.argmax(-1).item())
    lg = logits / temp
    if rep_pen > 1.0:
        seen = torch.tensor(ids, dtype=torch.long, device=logits.device)
        lg[seen] = torch.where(lg[seen] > 0, lg[seen] / rep_pen, lg[seen] * rep_pen)
    if top_k > 0:
        k = min(top_k, lg.size(-1))
        thr = lg.topk(k).values[-1]
        lg = torch.where(lg < thr, float('-inf'), lg)
    if top_p < 1.0:
        sorted_lg, idx = lg.sort(descending=True)
        sorted_p = F.softmax(sorted_lg, -1)
        sorted_lg[sorted_p.cumsum(-1) - sorted_p > top_p] = float('-inf')
        lg = lg.index_copy(0, idx, sorted_lg)
    p = F.softmax(lg, -1)
    return int(torch.multinomial(p, 1).item())

def stream(prompt_text, max_new, temp, top_k, top_p, rep_pen):
    ids = encode(tokenize(prompt_text))
    if len(ids) == 0:
        ids = [v2i.get('the', 2)]
    if len(ids) > MAX_CTX - max_new:
        ids = ids[-(MAX_CTX - max_new):]
        print('  (prompt truncated to fit context)')
    t0 = time.time()
    n = 0
    for _ in range(max_new):
        nxt = sample(next_logits(model, ids, dev), temp, top_k, top_p, rep_pen, ids)
        ids.append(nxt)
        n += 1
        piece = decode_piece(id2word[nxt] if nxt < len(id2word) else '?')
        if piece:
            print(piece, end='', flush=True)
    print(f'\n  [{n} tok | {n/(time.time()-t0):.0f} tok/s | temp={temp} topk={top_k} topp={top_p} rep={rep_pen}]', flush=True)

def help_text():
    print("""  /len N      max tokens per response (default 40, 1..256)
  /temp T     sampling temperature (0 = greedy, default 0.8)
  /topk K     top-k filter (0 = all, default 50)
  /topp P     top-p nucleus filter (1.0 = off, default 1.0)
  /rep R      repetition penalty, >1 penalizes tokens already in context (1.0 = off)
  /greedy     shortcut: temp 0
  /help       this message
  /quit       exit
  anything else is fed to the model as a prompt (fresh context each message)""")

def main():
    max_new, temp, top_k, top_p, rep_pen = 40, 0.8, 50, 1.0, 1.0
    print('\nType a prompt and press Enter. /help for commands.', flush=True)
    while True:
        try:
            line = input('\n>>> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == '/quit':
            break
        if line == '/help':
            help_text(); continue
        if line == '/greedy':
            temp, top_k = 0, 1
            print(f'  greedy mode (temp=0, topk=1)'); continue
        if line.startswith('/len'):
            max_new = max(1, min(256, int(line.split()[1])))
            print(f'  max_new = {max_new}'); continue
        if line.startswith('/temp'):
            temp = max(0.0, float(line.split()[1]))
            print(f'  temp = {temp}'); continue
        if line.startswith('/topk'):
            top_k = max(0, int(line.split()[1]))
            print(f'  topk = {top_k}'); continue
        if line.startswith('/topp'):
            top_p = min(1.0, max(0.01, float(line.split()[1])))
            print(f'  topp = {top_p}'); continue
        if line.startswith('/rep'):
            rep_pen = max(1.0, float(line.split()[1]))
            print(f'  rep = {rep_pen}'); continue
        stream(line, max_new, temp, top_k, top_p, rep_pen)

if __name__ == '__main__':
    torch.manual_seed(42)
    main()
