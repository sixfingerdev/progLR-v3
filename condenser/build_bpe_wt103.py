"""WikiText-103 → BPE tokenizer (50K) + tokenized chunks (seq=512).

Same pipeline as build_wt103_dataset.py (lowercase + whitespace pre-processing,
512-token chunks, <pad>=0 <unk>=1) but with a BPE tokenizer instead of word-level.
Outputs:
  condenser/wt103_data_bpe.pkl   (tr_x, tr_y, te_x, te_y, vocab_size)
  condenser/wt103_bpe_tokenizer.json  (tokenizers-format, for reuse)
"""
import os, pickle
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'wt103_raw')
SEQ = 512
VS = 50000

import pyarrow.parquet as pq

def load_text(split):
    fname = {'train': 'train-00000-of-00002.parquet',
             'train2': 'train-00001-of-00002.parquet',
             'valid': 'validation-00000-of-00001.parquet',
             'test': 'test-00000-of-00001.parquet'}[split]
    path = os.path.join(RAW, 'wikitext-103-v1', fname)
    t = pq.read_table(path)['text'].to_pylist()
    return '\n'.join(t)

print('Loading texts...', flush=True)
train_text = load_text('train') + load_text('train2')
valid_text = load_text('valid')
print(f'Train chars: {len(train_text):,}', flush=True)

print('Training BPE (50K, lowercase + whitespace)...', flush=True)
tok = Tokenizer(models.BPE(unk_token='<unk>'))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=VS, special_tokens=['<pad>', '<unk>'],
                              min_frequency=2, initial_alphabet=[])
tok.train_from_iterator([t.lower() for t in train_text.split('\n')], trainer)
print(f'BPE vocab size: {tok.get_vocab_size()}', flush=True)

with open(os.path.join(BASE, 'wt103_bpe_tokenizer.json'), 'w', encoding='utf-8') as f:
    f.write(tok.to_str())

def encode_fast(text):
    """Encode lines in C-speed batches, join with single spaces (like the word-level
    pipeline: re.sub(r'\\s+', ' ', ...) collapses line breaks to spaces)."""
    ids = []
    lines = text.lower().split('\n')
    BATCH = 5000
    for i in range(0, len(lines), BATCH):
        enc = tok.encode_batch(lines[i:i + BATCH], add_special_tokens=False)
        for e in enc:
            ids.extend(e.ids)
    return ids

print('Tokenizing train...', flush=True)
train_ids = encode_fast(train_text)
valid_ids = encode_fast(valid_text)
print(f'Train tokens: {len(train_ids):,}', flush=True)

def chunk(ids, name=''):
    xs, ys = [], []
    for i in range(0, len(ids) - SEQ, SEQ):
        xs.append(ids[i:i + SEQ])
        ys.append(ids[i + 1:i + SEQ + 1])
    print(f'{name}: {len(xs):,} chunks (of {len(ids):,} tokens)', flush=True)
    return xs, ys

tr_x, tr_y = chunk(train_ids, 'train')
te_x, te_y = chunk(valid_ids, 'valid')

data = {'tr_x': tr_x, 'tr_y': tr_y, 'te_x': te_x, 'te_y': te_y, 'vocab_size': VS}
with open(os.path.join(BASE, 'wt103_data_bpe.pkl'), 'wb') as f:
    pickle.dump(data, f)
print('Saved condenser/wt103_data_bpe.pkl', flush=True)
print('Done!', flush=True)
