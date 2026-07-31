"""Build aligned 250K vocab: old 50K ids preserved, 200K new words appended.
Old vocab order must NOT change so learned U rows transfer 1:1.
OOV words map to <unk> (id=1)."""
import os, re, pickle, time
from collections import Counter
import numpy as np
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'wt103_raw')
SEQ = 512
NEW_VS = 250000

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

# Load old vocab (word -> id, ids 0..49999)
with open(os.path.join(BASE, 'wt103_vocab.pkl'), 'rb') as f:
    old = pickle.load(f)
old_vocab = old['vocab']
OLD_VS = len(old_vocab)
print(f'Old vocab: {OLD_VS}', flush=True)

print('Loading text...', flush=True)
train_text = load_text('train') + load_text('train2')
valid_text = load_text('valid')
print('Tokenizing...', flush=True)
train_words = tokenize(train_text)
valid_words = tokenize(valid_text)
print(f'Train tokens: {len(train_words):,}', flush=True)

# Build new vocab: old order preserved, then 200K new words by frequency
print('Building 250K vocab...', flush=True)
counts = Counter(train_words)
new_vocab = dict(old_vocab)
next_id = OLD_VS
for w, _ in counts.most_common():
    if w in new_vocab:
        continue
    new_vocab[w] = next_id
    next_id += 1
    if next_id >= NEW_VS:
        break
print(f'New vocab: {len(new_vocab):,}', flush=True)

def chunk(words, name=''):
    ids = [new_vocab.get(w, 1) for w in words]  # OOV -> <unk>
    xs, ys = [], []
    for i in range(0, len(ids) - SEQ, SEQ):
        xs.append(ids[i:i+SEQ])
        ys.append(ids[i+1:i+SEQ+1])
    arr_x = np.array(xs, dtype=np.int32)
    arr_y = np.array(ys, dtype=np.int32)
    print(f'{name}: {arr_x.shape[0]:,} chunks', flush=True)
    return arr_x, arr_y

print('Chunking...', flush=True)
tr_x, tr_y = chunk(train_words, 'train')
te_x, te_y = chunk(valid_words, 'valid')

data = {'tr_x': tr_x, 'tr_y': tr_y, 'te_x': te_x, 'te_y': te_y,
        'vocab_size': NEW_VS}
with open(os.path.join(BASE, 'wt103_data_250k.pkl'), 'wb') as f:
    pickle.dump(data, f)
id2word = {v: k for k, v in new_vocab.items()}
with open(os.path.join(BASE, 'wt103_vocab_250k.pkl'), 'wb') as f:
    pickle.dump({'vocab': new_vocab, 'id2word': id2word}, f)
print(f'Saved. Train: {tr_x.shape}, Valid: {te_x.shape}', flush=True)
print('Done!', flush=True)
