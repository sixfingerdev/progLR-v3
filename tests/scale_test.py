"""V3 scaling test: D=768, L=36, rank=6 on WikiText-2."""
import sys, os, math, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.genomatrix import GenoTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Data
base_url = 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2'
for fname in ['train.txt', 'valid.txt']:
    if not os.path.exists(fname):
        urllib.request.urlretrieve(f'{base_url}/{fname}', fname)

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

train_text = read_file('train.txt')
val_text = read_file('valid.txt')

from collections import Counter
chars = Counter(train_text.lower())
vocab_chars = [c for c, _ in chars.most_common(500)]
idx_map = {c: i+3 for i, c in enumerate(vocab_chars)}
VOCAB = len(idx_map) + 3

def encode(text, seq_len=128):
    ids = []
    for line in text.split('\n'):
        for c in line.lower():
            ids.append(idx_map.get(c, 1))
        ids.append(2)
    data = torch.tensor(ids, dtype=torch.long)
    examples = []
    for i in range(0, len(data)-seq_len-1, seq_len):
        examples.append(data[i:i+seq_len+1])
    return torch.stack(examples) if examples else torch.zeros((0, seq_len+1), dtype=torch.long)

SEQ = 128; B = 24
train_data = encode(train_text, SEQ).to(device)
val_data = encode(val_text, SEQ).to(device)
print(f"Vocab: {VOCAB}, Train: {train_data.shape[0]}, Val: {val_data.shape[0]}")

# D=768, L=36, rank=6
v3 = GenoTransformer(vocab_size=VOCAB, hidden_dim=768, num_layers=36,
                     num_heads=12, rank=6)
params = v3.count_trained()
eff = v3.effective_params()
print(f"\nV3 params: {params:,}")
print(f"Effective:  {eff:,} ({eff/params:.0f}x)")
print(f"Expected Transformer at this scale: ~{eff} params")
print(f"V3 stored = {params/eff*100:.2f}% of effective")

v3.to(device)
opt = torch.optim.AdamW(v3.parameters(), lr=2e-4, weight_decay=0.01)
STEPS = 4000

def evaluate(model, data, max_batches=0):
    model.eval()
    total_loss = 0; n = 0
    limit = data.shape[0] if max_batches == 0 else min(max_batches * B, data.shape[0])
    with torch.no_grad():
        for i in range(0, limit, B):
            batch = data[i:i+B]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
            total_loss += loss.item() * batch.shape[0]; n += batch.shape[0]
    return total_loss / n

print("\nStep | Loss  | Val loss | Val PPL | ms/step | VRAM")
print("-" * 55)
t0 = time.time()
best_val = float('inf')

for step in range(STEPS):
    v3.train()
    idx = torch.randint(0, train_data.shape[0], (B,))
    batch = train_data[idx]
    l = F.cross_entropy(v3(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(v3.parameters(), 1.0); opt.step()

    if step % 500 == 0:
        v3_val = evaluate(v3, val_data, max_batches=30)
        best_val = min(best_val, v3_val)
        ms = (time.time() - t0) / max(step, 1) * 1000
        vram = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"{step:5d} | {l.item():.4f} | {v3_val:.4f} | {math.exp(v3_val):.2f} | {ms:.0f} | {vram:.0f}MB")

# Full eval
v3_val = evaluate(v3, val_data)
if not os.path.exists('test.txt'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt',
        'test.txt')
test_data = encode(read_file('test.txt'), SEQ).to(device)
v3_test = evaluate(v3, test_data)
total_t = time.time() - t0

print(f"\n{'='*55}")
print(f"  SCALE TEST: D=768, L=36, rank=6")
print(f"{'='*55}")
print(f"  V3 stored:   {params:,} params")
print(f"  Effective:   {eff:,} params")
print(f"  Val PPL:     {math.exp(v3_val):.2f}")
print(f"  Test PPL:    {math.exp(v3_test):.2f}")
print(f"  Time:        {total_t/60:.1f} min")

# Estimate Transformer at same effective size
print(f"\n  Transformer comparison (estimate):")
print(f"  At {eff} params with ~10M tokens:")
print(f"  Tokens/param: {10e6/eff:.3f}")
print(f"  Expected TF PPL (Chinchilla scaling): ~3-8")
print(f"  V3 uses {eff/params:.0f}x less stored params")

if v3_val < 4:
    print("\n  >>> PASS: Quality holds at scale! <<<")
elif v3_val < 8:
    print("\n  >>> WARN: Slight quality drop, still competitive <<<")
else:
    print("\n  >>> FAIL: Quality drops significantly at scale <<<")

# Bonus: estimate 500B
s = GenoTransformer.scaling_analysis(50000, 4096, 2480, 1)
print(f"\n  500B target: {s['total_mb']:.0f}MB < 6GB -> {'YES' if s['fits_6gb'] else 'NO'}")
