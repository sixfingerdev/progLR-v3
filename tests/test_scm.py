"""
SCM vs Transformer: Arithmetic Reasoning (a+b mod 100)
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.scm import SCM
from condenser.baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

NUM = 100
PLUS = NUM
EQUALS = NUM + 1
VOCAB = NUM + 2

SEQ = 4
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
        seqs.append([a, PLUS, b, EQUALS, c])
    return torch.tensor(seqs, dtype=torch.long, device=device)

train_data = make_data(train_idx)
test_data = make_data(test_idx)

B, STEPS = 64, 10000

configs = [
    ('SCM v3 (rank-2 route)', SCM(VOCAB, 128, 32, 8, 64)),
    ('Transformer', BaselineTransformer(VOCAB, 128, 4, 4)),
]

def evaluate(model, data):
    model.eval()
    total = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], min(B, data.shape[0])):
            b = data[i:i+min(B, data.shape[0])]
            logits = model(b[:, :4])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1))
            total += loss.item() * b.shape[0]; n += b.shape[0]
    return total / n

def accuracy(model, data):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], min(B, data.shape[0])):
            b = data[i:i+min(B, data.shape[0])]
            logits = model(b[:, :4])
            preds = logits.argmax(dim=-1)
            correct += (preds[:, -1] == b[:, -1]).sum().item()
            total += b.shape[0]
    return correct / total * 100

print(f'\n{"="*80}')
print(f'  ARITHMETIC REASONING: SCM v1 vs Transformer')
print(f'{"="*80}')
for name, model in configs:
    print(f'  {name:20}: {model.count_params():>8,} params ({model.count_params()*2/1e6:.3f}MB fp16)')

print(f'\n  {"Config":22} {"Step":>5} {"TrainLoss":>10} {"TestLoss":>9} {"OODAcc":>8} {"TestPPL":>8} {"Time":>5}')
print(f'  {"-"*72}')

results = {}
for name, model in configs:
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, train_data.shape[0], (B,))
        batch = train_data[idx]

        t0 = time.time()
        loss = F.cross_entropy(model(batch[:, :4]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        dt = time.time() - t0

        if step % 500 == 0:
            tl = loss.item()
            tst_loss = evaluate(model, test_data)
            tst_acc = accuracy(model, test_data)
            print(f'  {name:22} {step:5d} {tl:10.4f} {tst_loss:9.4f} {tst_acc:7.1f}% {math.exp(tst_loss):8.2f} {dt:5.2f}s', flush=True)

    test_loss = evaluate(model, test_data)
    test_acc = accuracy(model, test_data)
    vram = torch.cuda.max_memory_allocated(device) / 1e6
    results[name] = {'test_loss': test_loss, 'test_acc': test_acc, 'test_ppl': math.exp(test_loss), 'vram': vram}

print(f'\n{"="*80}')
print(f'  FINAL RESULTS')
print(f'{"="*80}')
print(f'  {"Config":22} {"Params":>8} {"TestLoss":>9} {"OODAcc":>8} {"TestPPL":>8} {"VRAM":>8}')
print(f'  {"-"*65}')
for name, model in configs:
    r = results[name]
    print(f'  {name:22} {model.count_params():>8,} {r["test_loss"]:9.4f} {r["test_acc"]:7.1f}% {r["test_ppl"]:8.2f} {r["vram"]:7.0f}MB')

trans_acc = results.get('Transformer', {}).get('test_acc', 0)
print(f'\n  VERDICT:')
for name, _ in configs:
    scm_acc = results.get(name, {}).get('test_acc', 0)
    if 'SCM' in name and scm_acc >= trans_acc * 0.9:
        print(f'  >>> {name} TRANSFORMER I YENDI! ({scm_acc:.1f}% vs TF={trans_acc:.1f}%) <<<')
    elif 'SCM' in name and scm_acc >= 50:
        print(f'  >>> {name} kismen ogreniyor ({scm_acc:.1f}% vs TF={trans_acc:.1f}%) <<<')
    elif 'SCM' in name:
        print(f'  >>> {name} basarisiz ({scm_acc:.1f}% vs TF={trans_acc:.1f}%) <<<')
print(f'  Random chance: {100/VOCAB:.1f}%')
