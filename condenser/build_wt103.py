"""Build WikiText-103 dataset for training."""
import os, re, pickle
from collections import Counter
from datasets import load_dataset

def build_dataset(vocab_size=50000, seq_len=512, cache_dir=None):
    print('Loading WikiText-103 from HuggingFace...', flush=True)
    ds = load_dataset('wikitext', 'wikitext-103-v1', split=['train', 'validation', 'test'],
                      cache_dir=cache_dir)
    train_text, valid_text, test_text = [d['text'] for d in ds]

    def tokenize(texts, max_tokens=None):
        all_words = []
        for t in texts:
            words = re.sub(r'\s+', ' ', t.strip().lower()).split()
            all_words.extend(words)
            if max_tokens and len(all_words) >= max_tokens:
                break
        return all_words[:max_tokens] if max_tokens else all_words

    print('Tokenizing train...', flush=True)
    train_words = tokenize(train_text)
    print(f'Train: {len(train_words):,} tokens', flush=True)

    # Build vocab
    counts = Counter(train_words)
    most_common = counts.most_common(vocab_size - 2)
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, (w, _) in enumerate(most_common):
        vocab[w] = i + 2
    vs = len(vocab)

    def chunk(words):
        ids = [vocab.get(w, 1) for w in words]
        xs = [ids[i:i+seq_len] for i in range(0, len(ids)-seq_len, seq_len)]
        ys = [ids[i+1:i+seq_len+1] for i in range(0, len(ids)-seq_len, seq_len)]
        return xs, ys

    print('Chunking train...', flush=True)
    tr_x, tr_y = chunk(train_words)
    print(f'Train chunks: {len(tr_x):,}', flush=True)

    print('Tokenizing valid...', flush=True)
    valid_words = tokenize(valid_text, 1500000)
    te_x, te_y = chunk(valid_words)
    print(f'Valid chunks: {len(te_x):,}', flush=True)

    data = {
        'tr_x': tr_x, 'tr_y': tr_y,
        'te_x': te_x, 'te_y': te_y,
        'vocab_size': vs, 'vocab': vocab,
    }
    dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wt103_data.pkl')
    with open(dump_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {dump_path}', flush=True)
    print(f'Vocab size: {vs}', flush=True)
    print(f'Train: {len(tr_x):,} sequences', flush=True)
    print(f'Valid: {len(te_x):,} sequences', flush=True)
    return data

if __name__ == '__main__':
    build_dataset(50000, 512)
