"""Does higher LoRA rank unlock reasoning? V3 arithmetic with rank=8,32,64 vs Transformer."""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.genomatrix import GenoTransformer, _R_POOL
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

NUM = 100
PLUS, EQUALS = NUM, NUM + 1
VOCAB = NUM + 2
SEQ = 4

pairs = [(a, b) for a in range(NUM) for b in range(NUM)]
c_sum = [(a + b) % NUM for a, b in pairs]

import random
random.seed(42)
indices = list(range(len(pairs)))
random.shuffle(indices)
split = int(len(indices) * 0.8)

def make_data(idx_list):
    seqs = []
    for idx in idx_list:
        a, b = pairs[idx]; c = c_sum[idx]
        seqs.append([a, PLUS, b, EQUALS, c])
    return torch.tensor(seqs, dtype=torch.long, device=device)

train_data = make_data(indices[:split])
test_data = make_data(indices[split:])
print(f'Data: train {train_data.shape[0]}, test {test_data.shape[0]}, vocab {VOCAB}')

D, L, HEADS = 128, 4, 4
B, STEPS = 64, 5000

configs = [
    (f'V3(r={r})', GenoTransformer(VOCAB, D, L, HEADS, rank=r, alpha=1.0)) for r in [8, 32, 64]
] + [('Transformer', BaselineTransformer(VOCAB, D, L, HEADS))]

def count_params(m):
    return m.count_trained() if hasattr(m, 'count_trained') else m.count_params()

def evaluate(model, data):
    model.eval(); total = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]
            logits = model(b[:, :4])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1))
            total += loss.item() * b.shape[0]; n += b.shape[0]
    return total / n

def accuracy(model, data):
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]
            logits = model(b[:, :4])
            preds = logits.argmax(dim=-1)
            correct += (preds[:, -1] == b[:, -1]).sum().item()
            total += b.shape[0]
    return correct / total * 100

print(f'  {"Config":15} {"Params":>8} {"Step":>5} {"TestLoss":>9} {"TestAcc":>8} {"TestPPL":>8}')
print(f'  {"-"*55}')

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

        if step % 1000 == 0:
            tst_loss = evaluate(model, test_data)
            tst_acc = accuracy(model, test_data)
            print(f'  {name:15} {count_params(model):>8,} {step:5d} {tst_loss:9.4f} {tst_acc:7.1f}% {math.exp(tst_loss):8.2f}')

    tst_loss = evaluate(model, test_data)
    tst_acc = accuracy(model, test_data)
    vram = torch.cuda.max_memory_allocated(device)/1e6
    results[name] = {'test_loss': tst_loss, 'test_acc': tst_acc, 'test_ppl': math.exp(tst_loss), 'vram': vram}

print(f'\n{"="*55}')
print(f'  FINAL: RANK vs REASONING')
print(f'{"="*55}')
print(f'  {"Config":15} {"Params":>8} {"TestAcc":>9} {"TestPPL":>9} {"VRAM":>7}')
print(f'  {"-"*45}')
for name, model in configs:
    r = results[name]
    print(f'  {name:15} {count_params(model):>8,} {r["test_acc"]:>8.1f}% {r["test_ppl"]:>9.2f} {r["vram"]:>6.0f}MB')
print(f'\n  Random chance: {1/VOCAB*100:.1f}%')
