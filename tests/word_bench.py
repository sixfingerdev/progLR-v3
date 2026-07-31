"""Word-level benchmark: V3 GenoTransformer vs Transformer on WikiText-2."""
import sys, os, math, time, re, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.genomatrix import GenoTransformer
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Download WikiText-2
base = 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2'
for f in ['train.txt', 'valid.txt', 'test.txt']:
    if not os.path.exists(f):
        urllib.request.urlretrieve(f'{base}/{f}', f)

def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Simple word tokenizer
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\'\.,!?;:]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

print("Tokenizing...")
train_text = read_text('train.txt')
val_text = read_text('valid.txt')
test_text = read_text('test.txt')

train_words = tokenize(train_text)
val_words = tokenize(val_text)
test_words = tokenize(test_text)

# Build vocab
from collections import Counter
word_counts = Counter(train_words)
vocab_list = ['<pad>', '<unk>', '<bos>', '<eos>'] + [w for w, _ in word_counts.most_common(5000)]
VOCAB = len(vocab_list)
w2i = {w: i for i, w in enumerate(vocab_list)}
print(f"Vocab: {VOCAB}")
print(f"Train words: {len(train_words):,}, Val: {len(val_words):,}, Test: {len(test_words):,}")

def encode(words, seq_len=64):
    ids = [w2i['<bos>']]
    for w in words:
        ids.append(w2i.get(w, w2i['<unk>']))
    ids.append(w2i['<eos>'])
    data = torch.tensor(ids, dtype=torch.long)
    examples = []
    for i in range(0, len(data)-seq_len-1, seq_len//2):
        examples.append(data[i:i+seq_len+1])
    return torch.stack(examples) if examples else torch.zeros((0, seq_len+1), dtype=torch.long)

SEQ = 64
BATCH = 32
train_data = encode(train_words, SEQ).to(device)
val_data = encode(val_words, SEQ).to(device)
test_data = encode(test_words, SEQ).to(device)
print(f"Train seqs: {train_data.shape[0]}, Val: {val_data.shape[0]}, Test: {test_data.shape[0]}")

D = 512
LAYERS = 12
HEADS = 8
RANK = 8
STEPS = 5000

v3 = GenoTransformer(vocab_size=VOCAB, hidden_dim=D, num_layers=LAYERS,
                     num_heads=HEADS, rank=RANK)
tf = BaselineTransformer(vocab_size=VOCAB, hidden_dim=D, num_layers=LAYERS,
                         num_heads=HEADS)

v3_p = v3.count_trained()
tf_p = tf.count_params()
print(f"\nV3 params: {v3_p:,}")
print(f"TF  params: {tf_p:,}")
print(f"Compression: {tf_p/max(v3_p,1):.1f}x")

v3.to(device); tf.to(device)

def evaluate(model, data, max_steps=0):
    model.eval()
    total_loss = 0; n = 0
    limit = data.shape[0] if max_steps == 0 else min(max_steps * BATCH, data.shape[0])
    with torch.no_grad():
        for i in range(0, limit, BATCH):
            b = data[i:i+BATCH]
            logits = model(b[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1))
            total_loss += loss.item() * b.shape[0]; n += b.shape[0]
    return total_loss / n

v3_opt = torch.optim.AdamW(v3.parameters(), lr=3e-4, weight_decay=0.01)
tf_opt = torch.optim.AdamW(tf.parameters(), lr=3e-4, weight_decay=0.01)

print("\nStep | V3 train | V3 val | V3 PPL | TF train | TF val | TF PPL")
print("-" * 70)
v3_best = float('inf'); tf_best = float('inf')

for step in range(STEPS):
    v3.train(); tf.train()
    idx = torch.randint(0, train_data.shape[0], (BATCH,))
    batch = train_data[idx]

    l = F.cross_entropy(v3(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
    v3_opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(v3.parameters(), 1.0); v3_opt.step()

    l = F.cross_entropy(tf(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
    tf_opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0); tf_opt.step()

    if step % 500 == 0:
        with torch.no_grad():
            v3_tr = F.cross_entropy(v3(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1)).item()
            tf_tr = F.cross_entropy(tf(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1)).item()
        v3_val = evaluate(v3, val_data, 30)
        tf_val = evaluate(tf, val_data, 30)
        v3_best = min(v3_best, v3_val); tf_best = min(tf_best, tf_val)
        print(f"{step:5d} | {v3_tr:.4f} {v3_val:.4f} {math.exp(v3_val):.2f} | {tf_tr:.4f} {tf_val:.4f} {math.exp(tf_val):.2f}")
        torch.cuda.empty_cache()

v3_test = evaluate(v3, test_data)
tf_test = evaluate(tf, test_data)

print(f"\n{'='*60}")
print(f"  WORD-LEVEL WIKITEXT-2: V3 vs TRANSFORMER")
print(f"{'='*60}")
print(f"  {'':24} {'V3':>12} {'TF':>12}")
print(f"  {'-'*50}")
print(f"  {'Params':24} {v3_p:>10,}  {tf_p:>10,}")
print(f"  {'Best Val PPL':24} {math.exp(v3_best):>12.2f} {math.exp(tf_best):>12.2f}")
print(f"  {'Test PPL':24} {math.exp(v3_test):>12.2f} {math.exp(tf_test):>12.2f}")
print(f"  {'Compression':24} {tf_p/max(v3_p,1):>11.1f}x {'1x':>12}")

gap = v3_test - tf_test
if gap < -0.5:
    print(f"\n  >>> V3 KAZANDI (gap={abs(gap):.2f} loss) <<<")
elif abs(gap) <= 0.5:
    print(f"\n  >>> BERABERE <<<")
else:
    print(f"\n  >>> Transformer kazandi (gap={abs(gap):.2f} loss) <<<")
