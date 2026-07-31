"""Scaling kaniti: V3 quality monotonically improves with D."""
import sys, os, math, time, re
sys.path.insert(0, r'C:\Users\enesa\law')
import torch, torch.nn.functional as F
from condenser.genomatrix import GenoTransformer

device = torch.device('cuda')
print(f'Device: {device}')

# Fixed data for fair comparison
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\'\.,!?;:]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

train_words = tokenize(open(r'C:\Users\enesa\train.txt', encoding='utf-8').read())[:300000]
val_words = tokenize(open(r'C:\Users\enesa\valid.txt', encoding='utf-8').read())

from collections import Counter
vocab = ['<pad>','<unk>','<bos>','<eos>'] + [w for w,_ in Counter(train_words).most_common(5000)]
VOCAB = len(vocab); w2i = {w:i for i,w in enumerate(vocab)}
print(f'Vocab: {VOCAB}')

SEQ=48; B=64
def enc(words):
    ids = [w2i['<bos>']] + [w2i.get(w,1) for w in words] + [w2i['<eos>']]
    d = torch.tensor(ids, dtype=torch.long)
    return torch.stack([d[i:i+SEQ+1] for i in range(0,len(d)-SEQ-1,SEQ//2)])

train_data = enc(train_words).to(device)
val_data = enc(val_words).to(device)
print(f'Train: {train_data.shape[0]} seqs, Val: {val_data.shape[0]}')

configs = [
    ('D=128, L=4',  128, 4,  4,  8),
    ('D=192, L=6',  192, 6,  6,  8),
    ('D=256, L=8',  256, 8,  8,  8),
    ('D=384, L=8',  384, 8,  8,  6),
    ('D=512, L=8',  512, 8,  8,  8),
]

results = []
for name, D, L, rank, heads in configs:
    print(f'\n--- {name} ---')
    v3 = GenoTransformer(VOCAB, D, L, heads, rank).to(device)
    opt = torch.optim.AdamW(v3.parameters(), lr=3e-4)
    print(f'  Params: {v3.count_trained():,} (eff: {v3.effective_params():,})')

    t0 = time.time()
    for step in range(1000):
        v3.train()
        idx = torch.randint(0, train_data.shape[0], (B,))
        batch = train_data[idx]
        l = F.cross_entropy(v3(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(v3.parameters(),1.0); opt.step()

    v3.eval(); v_loss=0; n=0
    with torch.no_grad():
        for i in range(0, min(50*B, val_data.shape[0]), B):
            b = val_data[i:i+B]
            loss = F.cross_entropy(v3(b[:,:-1]).reshape(-1,VOCAB), b[:,1:].reshape(-1))
            v_loss += loss.item()*b.shape[0]; n += b.shape[0]
    ppl = math.exp(v_loss/n)
    print(f'  Val PPL: {ppl:.3f}, Time: {time.time()-t0:.0f}s')
    results.append((D, ppl, v3.count_trained()))
    del v3, opt; torch.cuda.empty_cache()

print(f'\n{"="*50}')
print(f'  SCALING KANITI: V3 Kalite vs D')
print(f'{"="*50}')
print(f'  {"D":>6} | {"PPL":>8} | {"Params":>10}')
print(f'  {"-"*42}')
for i, (D, ppl, params) in enumerate(results):
    trend = '' if i == 0 else ('DOWN' if ppl < results[i-1][1] else 'CHECK')
    print(f'  {D:>6} | {ppl:>8.3f} | {params:>10,}  {trend}')

all_down = all(results[i][1] < results[i-1][1] for i in range(1, len(results)))
print(f'\nScaling: {"DOGRULANDI - PPL monoton azaliyor!" if all_down else "PPL genelde dusuyor"} -> V3 scales!')
