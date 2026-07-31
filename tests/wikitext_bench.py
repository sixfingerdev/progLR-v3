"""Real generalization test: same vocabulary, held-out patterns."""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from condenser.genomatrix import GenoTransformer
from baseline_transformer import BaselineTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

VOCAB = 2000
torch.manual_seed(42)

# Two different Markov chains over the SAME vocabulary
# Chain A (train): word[i+1] = (word[i] * 7 + 13) % VOCAB
# Chain B (val/test): word[i+1] = (word[i] * 3 + 7) % VOCAB
# Same vocab, different transition rules
# Model must learn "the transition follows A" vs "the transition follows B"
# This tests: can the model distinguish between structural patterns?

def chain_A(num_tokens):
    tokens = [torch.randint(0, VOCAB, (1,)).item()]
    for _ in range(num_tokens-1):
        tokens.append((tokens[-1] * 7 + 13) % VOCAB)
    return torch.tensor(tokens, dtype=torch.long)

def chain_B(num_tokens):
    tokens = [torch.randint(0, VOCAB, (1,)).item()]
    for _ in range(num_tokens-1):
        tokens.append((tokens[-1] * 3 + 7) % VOCAB)
    return torch.tensor(tokens, dtype=torch.long)

SEQ = 128

def chunk(data, stride=64):
    examples = []
    for i in range(0, len(data)-SEQ-1, stride):
        examples.append(data[i:i+SEQ+1])
    return torch.stack(examples) if examples else torch.zeros((0, SEQ+1), dtype=torch.long)

print("Generating data...")
train_tokens = chain_A(500000)
val_tokens_A = chain_B(50000)    # held-out pattern
val_tokens_B = chain_A(50000)    # seen pattern (to check memorization vs generalization)
test_tokens = chain_B(50000)

train_data = chunk(train_tokens).to(device)
val_data_A = chunk(val_tokens_A).to(device)
val_data_B = chunk(val_tokens_B).to(device)
test_data = chunk(test_tokens).to(device)

print(f"Train (Chain A): {train_data.shape[0]} seq")
print(f"Val (Chain A, seen):  {val_data_B.shape[0]} seq")
print(f"Val (Chain B, held-out): {val_data_A.shape[0]} seq")
print(f"Test (Chain B): {test_data.shape[0]} seq")

HIDDEN = 256
LAYERS = 6
HEADS = 4
RANK = 8
BATCH = 32
STEPS = 8000

v3 = GenoTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, num_layers=LAYERS,
                     num_heads=HEADS, rank=RANK)
tf = BaselineTransformer(vocab_size=VOCAB, hidden_dim=HIDDEN, num_layers=LAYERS,
                         num_heads=HEADS)

print(f"\nV3 params: {v3.count_trained():,}")
print(f"TF  params: {tf.count_params():,}")
print(f"Compression: {tf.count_params()/max(v3.count_trained(),1):.1f}x")
print(f"Effective: {v3.effective_params():,}")

v3.to(device); tf.to(device)

def evaluate(model, data):
    model.eval()
    total_loss = 0; n = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], BATCH):
            batch = data[i:i+BATCH]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
            total_loss += loss.item() * batch.shape[0]; n += batch.shape[0]
    return total_loss / n

v3_opt = torch.optim.AdamW(v3.parameters(), lr=3e-4, weight_decay=0.01)
tf_opt = torch.optim.AdamW(tf.parameters(), lr=3e-4, weight_decay=0.01)

print("\nStep | V3 trn | V3 valA(seen) | V3 valB(new) | TF trn | TF valA(seen) | TF valB(new)")
print("-" * 80)
for step in range(STEPS):
    v3.train(); tf.train()
    idx = torch.randint(0, train_data.shape[0], (BATCH,))
    batch = train_data[idx]

    l = F.cross_entropy(v3(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
    v3_opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(v3.parameters(), 1.0); v3_opt.step()

    l = F.cross_entropy(tf(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
    tf_opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0); tf_opt.step()

    if step % 1000 == 0:
        with torch.no_grad():
            v3_tr = F.cross_entropy(v3(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1)).item()
            tf_tr = F.cross_entropy(tf(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1)).item()
        v3_A = evaluate(v3, val_data_B)
        v3_B = evaluate(v3, val_data_A)
        tf_A = evaluate(tf, val_data_B)
        tf_B = evaluate(tf, val_data_A)
        print(f"{step:5d} | {v3_tr:.4f} {v3_A:.4f} {v3_B:.4f} | {tf_tr:.4f} {tf_A:.4f} {tf_B:.4f}")
        torch.cuda.empty_cache()

v3_test = evaluate(v3, test_data)
tf_test = evaluate(tf, test_data)

print(f"\n{'='*60}")
print(f"  GENERALIZATION: SAME VOCAB, HELD-OUT TRANSITION RULES")
print(f"{'='*60}")
print(f"  {'':24} {'V3':>12} {'TF':>12}")
print(f"  {'-'*50}")
print(f"  {'Params':24} {v3.count_trained():>10,}  {tf.count_params():>10,}")
print(f"  {'Val ChainA (seen)':24} {v3_A:>12.4f} {tf_A:>12.4f}")
print(f"  {'Val ChainB (new)':24} {v3_B:>12.4f} {tf_B:>12.4f}")
print(f"  {'Test ChainB (new)':24} {v3_test:>12.4f} {tf_test:>12.4f}")
print(f"  {'Compression':24} {tf.count_params()/max(v3.count_trained(),1):>11.1f}x {'1x':>12}")

gap = v3_test - tf_test
if gap < -0.1:
    print(f"\n  >>> GENOTRANSFORMER WINS (gap={abs(gap):.4f}) <<<")
elif abs(gap) <= 0.1:
    print(f"\n  >>> TIE <<<")
else:
    print(f"\n  >>> Transformer wins by {abs(gap):.4f} loss <<<")

# Key metric: generalization gap = val_chainB - val_chainA
# Shows how much worse the model is on the unseen pattern
print(f"\n  Generalization Gap (new pattern - seen pattern):")
print(f"  V3: {v3_B - v3_A:.4f}")
print(f"  TF: {tf_B - tf_A:.4f}")
