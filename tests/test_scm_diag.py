"""
SCM diagnostic: Can it route from position 0 to position 3?
Task: [A, SEP, B, SEP] -> predict [SEP, B, SEP, A]
Position 3 must route back to position 0 to copy A.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.scm import SCM
from condenser.baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

VOCAB = 50
SEP = VOCAB - 1

data = []
for _ in range(5000):
    a = torch.randint(0, VOCAB - 1, (1,))
    b = torch.randint(0, VOCAB - 1, (1,))
    data.append([a.item(), SEP, b.item(), SEP])
data = torch.tensor(data, dtype=torch.long, device=device)

import random
random.seed(42)
idx = list(range(len(data)))
random.shuffle(idx)
split = int(len(idx) * 0.8)
train_data = data[idx[:split]]
test_data = data[idx[split:]]
print(f'Vocab={VOCAB}, SEP={SEP}, Train={train_data.shape}, Test={test_data.shape}')

D, L, K = 128, 4, 64
B, STEPS = 64, 500

models = [
    ('SCM v3', SCM(VOCAB, D, 8, L, K)),
    ('Transformer', BaselineTransformer(VOCAB, D, L, 4)),
]

# Target: [A, SEP, B, SEP] -> [SEP, B, SEP, A]
def make_target(x):
    return torch.cat([x[:, 1:], x[:, :1]], dim=1)

def evaluate(model, data):
    model.eval()
    total = 0; correct_pos = [0]*3; total_pos = [0]*3
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]
            logits = model(b[:, :4])
            preds = logits.argmax(dim=-1)
            target = make_target(b)
            for p in range(3):
                correct_pos[p] += (preds[:, p] == target[:, p]).sum().item()
                total_pos[p] += b.shape[0]
    return [c/t*100 for c, t in zip(correct_pos, total_pos)]

print(f'\n{"="*70}')
print(f'  ROUTING TEST: [A, SEP, B, SEP] -> [SEP, B, SEP, A]')
print(f'{"="*70}')
for name, model in models:
    print(f'  {name}: {model.count_params():>6,} params')

print(f'\n  {"Model":15} {"Step":>5} {"Pos0(SEP)":>9} {"Pos1(B)":>8} {"Pos2(A)":>8} {"Time":>5}')
print(f'  {"-"*50}')

results = {}
for name, model in models:
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, train_data.shape[0], (B,))
        batch = train_data[idx]
        t0 = __import__('time').time()
        logits = model(batch[:, :4])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), make_target(batch).reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        dt = __import__('time').time() - t0

        if step % 100 == 0:
            accs = evaluate(model, test_data[:500])
            print(f'  {name:15} {step:5d} {accs[0]:8.1f}% {accs[1]:7.1f}% {accs[2]:7.1f}% {dt:5.2f}s', flush=True)

    accs = evaluate(model, test_data)
    vram = torch.cuda.max_memory_allocated(device) / 1e6
    results[name] = {'accs': accs, 'vram': vram}

print(f'\n{"="*70}')
print(f'  SUMMARY')
print(f'{"="*70}')
for name, r in sorted(results.items(), key=lambda x: -x[1]['accs'][2]):
    print(f'  {name:20}: SEP={r["accs"][0]:.1f}%  B={r["accs"][1]:.1f}%  A-copy={r["accs"][2]:.1f}%  VRAM={r["vram"]:.0f}MB')
print(f'  Random: {100/VOCAB:.1f}%')
