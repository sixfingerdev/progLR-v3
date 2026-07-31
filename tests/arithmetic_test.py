"""ARITHMETIC REASONING TEST: Does V3 learn rules or just memorize?
a + b = c (mod 100). Train on 80% of pairs, test on held-out 20%.
If model generalizes (low test PPL) → learning. If only memorizes (high test PPL) → n-gram.
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import torch
import torch.nn.functional as F
from condenser.genomatrix import GenoTransformer, _R_POOL
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# --- Generate arithmetic data ---
# Tokens: 0-99 = numbers, 100 = PLUS, 101 = EQUALS
# Sequence: "<a> <PLUS> <b> <EQUALS> <c>"  (4 tokens)
NUM = 100
PLUS = NUM
EQUALS = NUM + 1
VOCAB = NUM + 2  # 0-99 are number tokens

SEQ = 4
NUM_PAIRS = NUM * NUM  # 10,000

# Generate all pairs a,b in [0,99]
pairs = [(a, b) for a in range(NUM) for b in range(NUM)]
c_sum = [(a + b) % NUM for a, b in pairs]

import random
random.seed(42)
indices = list(range(len(pairs)))
random.shuffle(indices)

split = int(len(indices) * 0.8)
train_idx = indices[:split]
test_idx = indices[split:]

print(f'Total pairs: {len(pairs)}, Train: {len(train_idx)}, Test: {len(test_idx)}')

def make_data(idx_list):
    seqs = []
    for idx in idx_list:
        a, b = pairs[idx]
        c = c_sum[idx]
        seqs.append([a, PLUS, b, EQUALS, c])  # 5 tokens: input = [a,+,b,=], target = [c]
    return torch.tensor(seqs, dtype=torch.long, device=device)

train_data = make_data(train_idx)
test_data = make_data(test_idx)

print(f'Train seqs: {train_data.shape}, Test seqs: {test_data.shape}')

# --- Models ---
D, L, RANK, HEADS = 128, 4, 8, 4
B, STEPS = 64, 5000

configs = [
    ('V3', GenoTransformer(VOCAB, D, L, HEADS, RANK, alpha=1.0)),
    ('Transformer', BaselineTransformer(VOCAB, D, L, HEADS)),
]

def count_params(m):
    return m.count_trained() if hasattr(m, 'count_trained') else m.count_params()

def evaluate(model, data):
    model.eval()
    total = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]
            logits = model(b[:, :4])  # [a, PLUS, b, EQUALS]
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1))
            total += loss.item() * b.shape[0]; n += b.shape[0]
    return total / n

def accuracy(model, data):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]
            logits = model(b[:, :4])  # (B, 4, VOCAB)
            preds = logits.argmax(dim=-1)  # (B, 4)
            targets = b[:, 1:]  # (B, 4) = [PLUS, b, EQUALS, c]
            # We only care about the last token (c) being correct
            correct += (preds[:, -1] == targets[:, -1]).sum().item()
            total += b.shape[0]
    return correct / total * 100

print(f'\n{"="*65}')
print(f'  ARITHMETIC REASONING TEST: V3 vs Transformer')
print(f'{"="*65}')
for name, model in configs:
    print(f'  {name}: {count_params(model):>6,} params')

print(f'\n  {"Config":15} {"Step":>5} {"TrainLoss":>10} {"TestLoss":>9} {"TestAcc":>8} {"TestPPL":>8}')
print(f'  {"-"*60}')

results = {}
for name, model in configs:
    _R_POOL.clear()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, train_data.shape[0], (B,))
        batch = train_data[idx]

        loss = F.cross_entropy(model(batch[:, :4]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        if step % 500 == 0:
            tl = loss.item()
            tst_loss = evaluate(model, test_data)
            tst_acc = accuracy(model, test_data)
            print(f'  {name:15} {step:5d} {tl:10.4f} {tst_loss:9.4f} {tst_acc:7.1f}% {math.exp(tst_loss):8.2f}')

    test_loss = evaluate(model, test_data)
    test_acc = accuracy(model, test_data)
    vram = torch.cuda.max_memory_allocated(device)/1e6
    results[name] = {'test_loss': test_loss, 'test_acc': test_acc, 'test_ppl': math.exp(test_loss), 'vram': vram}

print(f'\n{"="*65}')
print(f'  FINAL ARITHMETIC REASONING RESULTS')
print(f'{"="*65}')
print(f'  {"Config":15} {"Params":>6} {"TestLoss":>9} {"TestAcc":>8} {"TestPPL":>8} {"VRAM":>8}')
print(f'  {"-"*55}')
for name, model in configs:
    r = results[name]
    print(f'  {name:15} {count_params(model):>6,} {r["test_loss"]:9.4f} {r["test_acc"]:7.1f}% {r["test_ppl"]:8.2f} {r["vram"]:7.0f}MB')

# Verdict
trans_acc = results.get('Transformer', {}).get('test_acc', 0)
v3_acc = results.get('V3', {}).get('test_acc', 0)

print(f'\n  VERDICT:')
if v3_acc >= trans_acc * 0.9:
    print(f'  >>> V3 gercekten ogreniyor! (V3={v3_acc:.1f}% vs TF={trans_acc:.1f}%) <<<')
elif v3_acc >= 50:
    print(f'  >>> V3 kismen ogreniyor ama n-gram egilimi var <<<')
else:
    print(f'  >>> V3 ezberliyor, ogrenmiyor (n-gram motoru) <<<')

print(f'\n  Random chance accuracy: {100/VOCAB:.1f}%')
