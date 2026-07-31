"""FINAL WORD-LEVEL BENCHMARK: V3 vs Transformer, WikiText-2 + correct test set."""
import sys, os, math, time, re, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.genomatrix import GenoTransformer
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\'\.,!?;:]', ' ', text)
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

D, L, RANK, HEADS = 512, 12, 8, 8
B, STEPS = 32, 5000

v3 = GenoTransformer(VOCAB, D, L, HEADS, RANK).to(device)
tf = BaselineTransformer(VOCAB, D, L, HEADS).to(device)

print(f'\nV3 params: {v3.count_trained():,} | TF params: {tf.count_params():,} | {tf.count_params()/max(v3.count_trained(),1):.1f}x')

v3_opt = torch.optim.AdamW(v3.parameters(), lr=3e-4, weight_decay=0.01)
tf_opt = torch.optim.AdamW(tf.parameters(), lr=3e-4, weight_decay=0.01)

def evaluate(model, data):
    model.eval(); total = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]
            loss = F.cross_entropy(model(b[:,:-1]).reshape(-1,VOCAB), b[:,1:].reshape(-1))
            total += loss.item() * b.shape[0]; n += b.shape[0]
    return total / n

print('\nStep | V3 tr | V3 val | V3 PPL | TF tr | TF val | TF PPL')
print('-' * 65)
t0 = time.time()
v3_best = float('inf'); tf_best = float('inf')

for step in range(STEPS):
    v3.train(); tf.train()
    idx = torch.randint(0, train_data.shape[0], (B,))
    batch = train_data[idx]

    l = F.cross_entropy(v3(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
    v3_opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(v3.parameters(), 1.0); v3_opt.step()

    l = F.cross_entropy(tf(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
    tf_opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0); tf_opt.step()

    if step % 500 == 0 and step > 0:
        with torch.no_grad():
            v3_tr = F.cross_entropy(v3(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1)).item()
            tf_tr = F.cross_entropy(tf(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1)).item()
        v3_val = evaluate(v3, val_data); v3_best = min(v3_best, v3_val)
        tf_val = evaluate(tf, val_data); tf_best = min(tf_best, tf_val)
        print(f'{step:5d} | {v3_tr:.4f} {v3_val:.4f} {math.exp(v3_val):.2f} | {tf_tr:.4f} {tf_val:.4f} {math.exp(tf_val):.2f}')

v3_test = evaluate(v3, test_data)
tf_test = evaluate(tf, test_data)

print(f'\n{"="*60}')
print(f'  FINAL WORD-LEVEL: V3 vs TRANSFORMER')
print(f'{"="*60}')
print(f'  {"":24} {"V3":>12} {"TF":>12}')
print(f'  {"-"*50}')
print(f'  {"Params":24} {v3.count_trained():>10,}  {tf.count_params():>10,}')
print(f'  {"Best Val PPL":24} {math.exp(v3_best):>12.2f} {math.exp(tf_best):>12.2f}')
print(f'  {"Test PPL":24} {math.exp(v3_test):>12.2f} {math.exp(tf_test):>12.2f}')
print(f'  {"Compression":24} {tf.count_params()/max(v3.count_trained(),1):>11.1f}x {"1x":>12}')
print(f'  {"Time":24} {(time.time()-t0)/60:>11.1f}min {"":>12}')

gap = v3_test - tf_test
if gap < -0.5:
    print(f'\n  >>> V3 KAZANDI (gap={abs(gap):.2f}) <<<')
elif abs(gap) <= 0.5:
    print(f'  >>> BERABERE <<<')
else:
    print(f'  >>> Transformer kazandi (gap={abs(gap):.2f}) <<<')
