"""Hizli word-level test: 5-10 dk."""
import sys, os, math, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from condenser.genomatrix import GenoTransformer
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\'\.,!?;:]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

print('Loading data...')
train_words = tokenize(open(r'C:\Users\enesa\train.txt', encoding='utf-8').read())[:200000]
val_words = tokenize(open(r'C:\Users\enesa\valid.txt', encoding='utf-8').read())[:50000]
test_words = tokenize(open(r'C:\Users\enesa\law\tests\wiki.test.tokens', encoding='utf-8').read())[:50000]

from collections import Counter
vocab = ['<pad>','<unk>','<bos>','<eos>'] + [w for w,_ in Counter(train_words).most_common(3000)]
VOCAB = len(vocab); w2i = {w:i for i,w in enumerate(vocab)}
print(f'Vocab: {VOCAB}, Train: {len(train_words):,}')

SEQ = 48; B = 64
def enc(words):
    ids = [w2i['<bos>']] + [w2i.get(w,1) for w in words] + [w2i['<eos>']]
    d = torch.tensor(ids, dtype=torch.long)
    return torch.stack([d[i:i+SEQ+1] for i in range(0,len(d)-SEQ-1,SEQ//2)])

train_data = enc(train_words).to(device)
val_data = enc(val_words).to(device)
test_data = enc(test_words).to(device)
print(f'Train seqs: {train_data.shape[0]}, Val: {val_data.shape[0]}')

D, L, R, H = 384, 8, 8, 6
v3 = GenoTransformer(VOCAB, D, L, H, R).to(device)
tf = BaselineTransformer(VOCAB, D, L, H).to(device)
print(f'V3: {v3.count_trained():,} | TF: {tf.count_params():,} | {tf.count_params()/max(v3.count_trained(),1):.1f}x')

v3_opt = torch.optim.AdamW(v3.parameters(), lr=3e-4)
tf_opt = torch.optim.AdamW(tf.parameters(), lr=3e-4)

def eval_mod(m, data):
    m.eval(); t=0; n=0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]; l = F.cross_entropy(m(b[:,:-1]).reshape(-1,VOCAB), b[:,1:].reshape(-1))
            t += l.item()*b.shape[0]; n += b.shape[0]
    return t/n

print(f'\nStep | V3 tr | V3 val | PPL | TF tr | TF val | PPL')
print('-'*55)
t0 = time.time()
for step in range(1500):
    v3.train(); tf.train()
    idx = torch.randint(0, train_data.shape[0], (B,)); batch = train_data[idx]
    l = F.cross_entropy(v3(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
    v3_opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(v3.parameters(),1.0); v3_opt.step()
    l = F.cross_entropy(tf(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
    tf_opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(tf.parameters(),1.0); tf_opt.step()
    if step % 500 == 0:
        with torch.no_grad():
            vt = F.cross_entropy(v3(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1)).item()
            tt = F.cross_entropy(tf(batch[:,:-1]).reshape(-1,VOCAB), batch[:,1:].reshape(-1)).item()
        vv = eval_mod(v3, val_data); tv = eval_mod(tf, val_data)
        print(f'{step:4d} | {vt:.3f} {vv:.3f} {math.exp(vv):.1f} | {tt:.3f} {tv:.3f} {math.exp(tv):.1f}')

vt = eval_mod(v3, test_data); tt = eval_mod(tf, test_data)
print(f'\nTEST PPL: V3={math.exp(vt):.2f} vs TF={math.exp(tt):.2f}')
print(f'KAZANAN: {"V3" if vt < tt else "TF"} (gap={abs(vt-tt):.2f})')
print(f'{time.time()-t0:.0f}s')
