"""Extract vocab mapping from WikiText-103 raw data."""
import os, re, pickle
from collections import Counter
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'wt103_raw')

def load_text(split):
    fname = {'train': 'train-00000-of-00002.parquet',
             'train2': 'train-00001-of-00002.parquet',
             'valid': 'validation-00000-of-00001.parquet'}[split]
    path = os.path.join(RAW, 'wikitext-103-v1', fname)
    t = pq.read_table(path)['text'].to_pylist()
    return '\n'.join(t)

def tokenize(text, max_tokens=None):
    words = re.sub(r'\s+', ' ', text.strip().lower()).split()
    return words[:max_tokens] if max_tokens else words

train_text = load_text('train') + load_text('train2')
train_words = tokenize(train_text)
counts = Counter(train_words)
vocab = {'<pad>': 0, '<unk>': 1}
i = 2
for w, _ in counts.most_common(50000):
    if w not in vocab:
        vocab[w] = i; i += 1
        if i >= 50000: break

# Save both directions
id2word = {v: k for k, v in vocab.items()}
with open(os.path.join(BASE, 'wt103_vocab.pkl'), 'wb') as f:
    pickle.dump({'vocab': vocab, 'id2word': id2word}, f)
print(f'Vocab saved: {len(vocab):,} tokens')
