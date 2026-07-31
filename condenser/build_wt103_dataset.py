"""Process WikiText-103 parquet → tokenized chunks (vocab=50K, seq=512)."""
import os, re, pickle, sys
from collections import Counter
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'wt103_raw')
SEQ = 512

def load_text(split):
    fname = {'train': 'train-00000-of-00002.parquet',
             'train2': 'train-00001-of-00002.parquet',
             'valid': 'validation-00000-of-00001.parquet',
             'test': 'test-00000-of-00001.parquet'}[split]
    path = os.path.join(RAW, 'wikitext-103-v1', fname)
    t = pq.read_table(path)['text'].to_pylist()
    return '\n'.join(t)

def tokenize(text, max_tokens=None):
    words = re.sub(r'\s+', ' ', text.strip().lower()).split()
    return words[:max_tokens] if max_tokens else words

print('Loading texts...', flush=True)
train_text = load_text('train') + load_text('train2')
valid_text = load_text('valid')
print(f'Train chars: {len(train_text):,}', flush=True)
print(f'Valid chars: {len(valid_text):,}', flush=True)

print('Tokenizing train...', flush=True)
train_words = tokenize(train_text)
print(f'Train tokens: {len(train_words):,}', flush=True)

# Build vocab
print('Building vocab (50K)...', flush=True)
counts = Counter(train_words)
vocab = {'<pad>': 0, '<unk>': 1}
i = 2
for w, _ in counts.most_common(50000):
    if w not in vocab:
        vocab[w] = i
        i += 1
        if i >= 50000: break
VS = len(vocab)
print(f'Vocab size: {VS}', flush=True)

def chunk(words, name=''):
    ids = [vocab.get(w, 1) for w in words]
    xs, ys = [], []
    for i in range(0, len(ids) - SEQ, SEQ):
        xs.append(ids[i:i+SEQ])
        ys.append(ids[i+1:i+SEQ+1])
    print(f'{name}: {len(xs):,} chunks (of {len(ids):,} tokens)', flush=True)
    return xs, ys

print('Chunking...', flush=True)
tr_x, tr_y = chunk(train_words, 'train')
te_x, te_y = chunk(tokenize(valid_text), 'valid')

data = {'tr_x': tr_x, 'tr_y': tr_y, 'te_x': te_x, 'te_y': te_y, 'vocab_size': VS}
path = os.path.join(BASE, 'wt103_data.pkl')
with open(path, 'wb') as f:
    pickle.dump(data, f)
print(f'Saved to {path}', flush=True)
print(f'Train: {len(tr_x):,} seq, Valid: {len(te_x):,} seq', flush=True)
print('Done!', flush=True)
