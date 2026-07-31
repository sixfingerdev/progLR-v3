"""
Phase 1: WikiText-2 benchmark.
V3-Embed + 2 TF vs Standard Embed + 2 TF.
Compare: params, VRAM, perplexity.
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

from condenser.v3_embed import V3WeightEmbedding
from condenser.baseline_transformer import BaselineAttention, BaselineTransformerLayer


# --- Tokenizer ---
def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if len(line.strip()) > 0]

def build_vocab(texts, max_vocab=5000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(max_vocab - 2)
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab

def encode(text, vocab):
    return [vocab.get(w, vocab['<unk>']) for w in text.split()]

def encode_all(texts, vocab):
    ids = []
    for text in texts:
        ids.extend(encode(text, vocab))
    return ids


# --- Models ---
class StandardEmbeddingTF(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([
            BaselineTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos[:, :T]
        for layer in self.layers: x = layer(x)
        return self.head(self.norm(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class V3EmbeddingTF(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, rank=16, alpha=64):
        super().__init__()
        self.v3_embed = V3WeightEmbedding(vocab_size, hidden_dim, rank, alpha)
        self.pos = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([
            BaselineTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.v3_embed.forward_embed(input_ids) + self.pos[:, :T]
        for layer in self.layers: x = layer(x)
        return self.v3_embed.forward_head(self.norm(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def evaluate(model, data, batch_size, seq_len, vocab_size):
    model.eval()
    total_loss = 0.0; total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(data) - seq_len, seq_len):
            if i + batch_size * seq_len > len(data):
                break
            x = torch.stack([torch.tensor(data[j:j+seq_len], dtype=torch.long) for j in range(i, min(i+batch_size*seq_len, len(data)-seq_len), seq_len)]).cuda()
            if len(x) == 0: break
            y = torch.stack([torch.tensor(data[j+1:j+seq_len+1], dtype=torch.long) for j in range(i, min(i+batch_size*seq_len, len(data)-seq_len), seq_len)]).cuda()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


# --- Main ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    DATA_DIR = r'C:\Users\enesa\AppData\Local\Temp\opencode'
    train_texts = load_text(os.path.join(DATA_DIR, 'wikitext2_train.txt'))
    test_texts = load_text(os.path.join(DATA_DIR, 'wikitext2_test.txt'))

    print(f'Train lines: {len(train_texts)}', flush=True)
    print(f'Test lines: {len(test_texts)}', flush=True)

    VOCAB_SIZE = 5000
    vocab = build_vocab(train_texts, VOCAB_SIZE)
    VOCAB_SIZE = len(vocab)
    print(f'Vocab size: {VOCAB_SIZE}', flush=True)

    train_ids = encode_all(train_texts, vocab)
    test_ids = encode_all(test_texts, vocab)
    print(f'Train tokens: {len(train_ids):,}', flush=True)
    print(f'Test tokens: {len(test_ids):,}', flush=True)

    D = 256; L = 2; HEADS = 4; RANK = 16; ALPHA = 64
    B = 16; SEQ = 128; STEPS = 5000

    standard = StandardEmbeddingTF(VOCAB_SIZE, D, L, HEADS)
    v3_model = V3EmbeddingTF(VOCAB_SIZE, D, L, HEADS, RANK, ALPHA)

    print(f'\n{"Model":<25} {"Params":>12}')
    print(f'{"-"*40}')
    for name, m in [('Standard Embed+TF', standard), ('V3-Embed+TF', v3_model)]:
        print(f'  {name:<23} {m.count_params():>12,}', flush=True)

    def train_model(model, name, train_ids):
        from condenser.genomatrix import _R_POOL; _R_POOL.clear()
        torch.cuda.reset_peak_memory_stats(device)
        model.cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        best_ppl = float('inf')
        step_times = []

        print(f'\n{"="*55}', flush=True)
        print(f'  {name} ({model.count_params():,} params)', flush=True)
        print(f'{"="*55}', flush=True)

        for step in range(STEPS):
            t0 = time.time()
            model.train()
            idx = torch.randint(0, len(train_ids) - SEQ - 1, (B,))
            x = torch.stack([torch.tensor(train_ids[i:i+SEQ], dtype=torch.long) for i in idx]).cuda()
            y = torch.stack([torch.tensor(train_ids[i+1:i+SEQ+1], dtype=torch.long) for i in idx]).cuda()
            loss = F.cross_entropy(model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step_times.append(time.time() - t0)

            if step % 1000 == 0:
                val_loss = evaluate(model, test_ids, B, SEQ, VOCAB_SIZE)
                val_ppl = math.exp(min(val_loss, 12))
                best_ppl = min(best_ppl, val_ppl)
                vram = torch.cuda.memory_allocated(device) / 1e6
                avg_ms = sum(step_times[-100:]) / max(1, len(step_times[-100:])) * 1000
                print(f'  step={step:>4} | loss={loss.item():.3f} | val_ppl={val_ppl:.2f} | vram={vram:.0f}MB | {avg_ms:.0f}ms', flush=True)

        val_loss = evaluate(model, test_ids, B, SEQ, VOCAB_SIZE)
        val_ppl = math.exp(min(val_loss, 12))
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e6
        avg_ms = sum(step_times) / len(step_times) * 1000
        return {'ppl': val_ppl, 'best_ppl': best_ppl, 'vram': peak_vram, 'step_ms': avg_ms, 'params': model.count_params()}

    results = {}
    results['Standard'] = train_model(standard, 'Standard', train_ids)
    results['V3-Embed'] = train_model(v3_model, 'V3-Embed+TF', train_ids)

    print(f'\n{"="*60}')
    print(f'  WIKITEXT-2 BENCHMARK')
    print(f'{"="*60}')
    print(f'  {"Metric":<20} {"Standard":<18} {"V3-Embed":<18}')
    print(f'  {"-"*56}')
    for metric in ['params', 'best_ppl', 'ppl', 'vram', 'step_ms']:
        s = results['Standard'].get(metric, 0)
        v = results['V3-Embed'].get(metric, 0)
        if metric == 'params':
            print(f'  {metric:<20} {s/1000:>14.1f}K {v/1000:>14.1f}K', flush=True)
        elif metric == 'vram':
            print(f'  {metric:<20} {s:>14.0f}MB {v:>14.0f}MB', flush=True)
        elif metric == 'step_ms':
            print(f'  {metric:<20} {s:>14.1f} {v:>14.1f}', flush=True)
        else:
            print(f'  {metric:<20} {s:>14.2f} {v:>14.2f}', flush=True)

    savings = results['Standard']['params'] - results['V3-Embed']['params']
    print(f'\n  Embedding compression: {savings/1000:.1f}K params saved', flush=True)
    vram_savings = results['Standard']['vram'] - results['V3-Embed']['vram']
    print(f'  VRAM savings: {vram_savings:.0f}MB', flush=True)
    ppl_diff = results['V3-Embed']['best_ppl'] - results['Standard']['best_ppl']
    if ppl_diff < 0:
        print(f'  >>> V3-Embed WINS by {-ppl_diff:.2f} PPL!', flush=True)
    elif ppl_diff < 1:
        print(f'  >>> TIE (gap={ppl_diff:.2f} PPL)', flush=True)
    else:
        print(f'  Standard leads by {ppl_diff:.2f} PPL', flush=True)
