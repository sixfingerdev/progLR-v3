"""STEERING BENCHMARK: V3 (QR + alpha scaling) vs Transformer, WikiText-2 word level."""
import sys, os, math, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.genomatrix import GenoTransformer, generate_R_batch, _R_POOL
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def tokenize(text):
    text = re.sub(r'[^a-z0-9\s\'\.,!?;:]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

print("Tokenizing...")
train_words = tokenize(read_text(r'C:\Users\enesa\train.txt'))
val_words = tokenize(read_text(r'C:\Users\enesa\valid.txt'))
test_words = tokenize(read_text(r'C:\Users\enesa\law\tests\wiki.test.tokens'))
print(f'Train: {len(train_words):,}, Val: {len(val_words):,}, Test: {len(test_words):,}')

from collections import Counter
vocab_list = ['<pad>', '<unk>', '<bos>', '<eos>'] + [w for w, _ in Counter(train_words).most_common(5000)]
VOCAB = len(vocab_list)
w2i = {w: i for i, w in enumerate(vocab_list)}
print(f'Vocab: {VOCAB}')

SEQ = 64
def encode(words):
    ids = [w2i['<bos>']] + [w2i.get(w, w2i['<unk>']) for w in words] + [w2i['<eos>']]
    data = torch.tensor(ids, dtype=torch.long)
    examples = []
    for i in range(0, len(data)-SEQ-1, SEQ//2):
        examples.append(data[i:i+SEQ+1])
    return torch.stack(examples) if examples else torch.zeros((0, SEQ+1), dtype=torch.long)

train_data = encode(train_words).to(device)
val_data = encode(val_words).to(device)
test_data = encode(test_words).to(device)
print(f'Train seqs: {train_data.shape[0]}, Val: {val_data.shape[0]}, Test: {test_data.shape[0]}')

D, L, RANK, HEADS = 384, 6, 8, 6
B, STEPS = 32, 3000

# We need two versions: one with old R (randn), one with new R (QR)
# Since generate_R_batch is globally defined, we can't easily swap.
# Instead: save a copy of the original _orthogonal_matrix and generate_R_batch

# Actually, let me create a modified GenoTransformer that uses old R approach
# by overriding the layer's forward method to call a non-QR generate_R

import types

# Save old R generation for comparison
def randn_generate_R_batch(seed, D, device, dtype):
    total = 12 * D * D
    gen = torch.Generator(device=device)
    gen.manual_seed(abs(seed) % (2**31 - 1))
    all_R = torch.randn(total, generator=gen, device=device, dtype=dtype)
    scale = math.sqrt(D)
    off = 0
    s = D * D
    R_Q = all_R[off:off+s].view(D, D).div_(scale); off += s
    R_K = all_R[off:off+s].view(D, D).div_(scale); off += s
    R_V = all_R[off:off+s].view(D, D).div_(scale); off += s
    R_O = all_R[off:off+s].view(D, D).div_(scale); off += s
    s = 4 * D * D
    R_FF1 = all_R[off:off+s].view(4*D, D).div_(scale); off += s
    R_FF2 = all_R[off:off+s].view(D, 4*D).div_(scale)
    return {'q': R_Q, 'k': R_K, 'v': R_V, 'o': R_O, 'ff1': R_FF1, 'ff2': R_FF2}

_RAND_POOL = {}
class GenoTransformerOldR(GenoTransformer):
    """V3 with old torch.randn R (no QR), for comparison."""
    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(self._old_layer_forward, layer, x, use_reentrant=True)
        x = self.norm(x)
        return self.lm_head(x)
    
    @staticmethod
    def _old_layer_forward(layer, x):
        D = x.shape[-1]
        key = (layer.seed + layer.layer_id * 6, D)
        if key not in _RAND_POOL or _RAND_POOL[key]['q'].device != x.device or _RAND_POOL[key]['q'].dtype != x.dtype:
            _RAND_POOL[key] = randn_generate_R_batch(*key, x.device, x.dtype)
        Rs = _RAND_POOL[key]
        x = x + layer.attn(layer.norm1(x), Rs)
        x = x + layer.ffn(layer.norm2(x), Rs)
        return x

configs = [
    ("V3(OldR,a=1)", GenoTransformerOldR(VOCAB, D, L, HEADS, RANK, alpha=1.0)),
    ("V3(OldR,a=16)", GenoTransformerOldR(VOCAB, D, L, HEADS, RANK, alpha=16.0)),
    ("V3(QR,a=1)", GenoTransformer(VOCAB, D, L, HEADS, RANK, alpha=1.0)),
    ("V3(QR,a=16)", GenoTransformer(VOCAB, D, L, HEADS, RANK, alpha=16.0)),
    ("Transformer", BaselineTransformer(VOCAB, D, L, HEADS)),
]

def count_params(m):
    if hasattr(m, 'count_trained'):
        return m.count_trained()
    return m.count_params()

def evaluate(model, data):
    model.eval(); total = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]
            loss = F.cross_entropy(model(b[:,:-1]).reshape(-1,VOCAB), b[:,1:].reshape(-1))
            total += loss.item() * b.shape[0]; n += b.shape[0]
    return total / n

def lora_norm_ratio(model, force_randn=False):
    if not hasattr(model, 'layers'):
        return 0.0
    ratios = []
    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            seed = 42 + li * 6
            D_ = model.hidden_dim
            Rs = randn_generate_R_batch(seed, D_, device, torch.float32)
            for mod_name, mod in [('attn', layer.attn), ('ffn', layer.ffn)]:
                for op in ['Q','K','V','O']:
                    u = getattr(mod, f'U_{op}', None)
                    v = getattr(mod, f'V_{op}', None)
                    if u is None:
                        continue
                    r_key = op.lower()
                    if r_key in Rs:
                        lora_n = (u.to(torch.float32) @ v.to(torch.float32).T).norm().item()
                        r_n = Rs[r_key].norm().item()
                        if r_n > 0:
                            ratios.append(lora_n / r_n * 100)
                for op in ['FF1','FF2']:
                    u = getattr(mod, f'U_{op}', None)
                    v = getattr(mod, f'V_{op}', None)
                    if u is None:
                        continue
                    r_key = op.lower()
                    if r_key in Rs:
                        lora_n = (u.to(torch.float32) @ v.to(torch.float32).T).norm().item()
                        r_n = Rs[r_key].norm().item()
                        if r_n > 0:
                            ratios.append(lora_n / r_n * 100)
    return sum(ratios)/len(ratios) if ratios else 0.0

print()
print(f'{"="*80}')
print(f'  STEERING BENCHMARK: QR vs OldR, alpha vs no-alpha (D={D}, L={L})')
print(f'{"="*80}')
for name, model in configs:
    p = count_params(model)
    print(f'  {name:20}: {p:>8,} params (effective: {model.effective_params():,})' if 'V3' in name else f'  {name:20}: {p:>8,} params')

print()
print(f'  {"Config":20} {"Step":>5} {"VRAM":>7} {"ms/step":>9} {"TrLoss":>7} {"ValPPL":>7} {"LoRA/R%":>8}')
print(f'  {"-"*70}')

results = {}
for name, model in configs:
    torch.cuda.reset_peak_memory_stats(device)
    _R_POOL.clear()
    _RAND_POOL.clear()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_val = float('inf')
    times = []

    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, train_data.shape[0], (B,))
        batch = train_data[idx]

        t0 = time.time()
        loss = F.cross_entropy(model(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        if step % 500 == 0:
            val_loss = evaluate(model, val_data)
            best_val = min(best_val, val_loss)
            ratio = lora_norm_ratio(model) if 'V3' in name else 0
            vram = torch.cuda.max_memory_allocated(device)/1e6
            avg_t = sum(times[-100:])/max(len(times[-100:]),1)*1000
            print(f'  {name:20} {step:5d} {vram:6.0f}MB {avg_t:8.1f} {loss.item():7.4f} {math.exp(val_loss):7.2f} {ratio:8.2f}')
            torch.cuda.empty_cache()

    val_loss = evaluate(model, val_data)
    test_loss = evaluate(model, test_data)
    ratio = lora_norm_ratio(model) if 'V3' in name else 0
    vram = torch.cuda.max_memory_allocated(device)/1e6
    avg_t = sum(times[-500:])/max(len(times[-500:]),1)*1000 if len(times) >= 500 else sum(times)/len(times)*1000
    results[name] = {'vram': vram, 'time': avg_t, 'val_ppl': math.exp(val_loss), 'test_ppl': math.exp(test_loss), 'lora_ratio': ratio}

print()
print(f'{"="*80}')
print(f'  FINAL RESULTS')
print(f'{"="*80}')
print(f'  {"Config":20} {"Params":>8} {"VRAM":>8} {"ms/step":>9} {"ValPPL":>8} {"TestPPL":>8} {"LoRA/R%":>8}')
print(f'  {"-"*70}')
for name, model in configs:
    r = results[name]
    p = count_params(model)
    print(f'  {name:20} {p:>8,} {r["vram"]:>7.0f}MB {r["time"]:>8.1f} {r["val_ppl"]:>8.2f} {r["test_ppl"]:>8.2f} {r["lora_ratio"]:>8.2f}')

# Key comparison: alpha effect with old R
if 'V3(OldR,a=1)' in results and 'V3(OldR,a=16)' in results:
    g = results['V3(OldR,a=1)']['test_ppl'] - results['V3(OldR,a=16)']['test_ppl']
    print(f'\n  OldR: alpha=1 vs alpha=16 gap: {abs(g):.2f} PPL ({"alpha wins" if g>0 else "no effect"})')

# Key comparison: QR vs OldR with alpha=16
if 'V3(QR,a=16)' in results and 'V3(OldR,a=16)' in results:
    g = results['V3(QR,a=16)']['test_ppl'] - results['V3(OldR,a=16)']['test_ppl']
    print(f'  a=16: QR vs OldR gap: {abs(g):.2f} PPL ({"QR wins" if g<0 else "OldR wins"})')

# LoRA/R ratio effect
for n, r in results.items():
    if 'V3' in n:
        print(f'  {n}: LoRA/R={r["lora_ratio"]:.1f}%, TestPPL={r["test_ppl"]:.2f}')
