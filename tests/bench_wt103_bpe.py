"""
WikiText-103 BPE Benchmark: V3 GenoTransformer vs Baseline Transformer
GPT-2-small scale: D=768, L=12, heads=12
"""
import sys, time, math, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from condenser.genomatrix import GenoTransformer
from baseline_transformer import BaselineTransformer

device = torch.device("cuda")
SEQ = 128
B = 8
D = 768
L = 12
HEADS = 12
RANK = 6
STEPS = 4000

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
VOCAB = tokenizer.vocab_size

print(f"V3 GenoTransformer (D={D}, L={L}, rank={RANK}) vs Baseline Transformer (D={D}, L={L})")
print(f"Dataset: WikiText-103 BPE (vocab={VOCAB})")
print(f"Batch={B}, Seq={SEQ}, Steps={STEPS}")

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
texts = []
N_CHUNKS = 200000
for i, example in enumerate(ds):
    if i >= N_CHUNKS:
        break
    t = example["text"].strip()
    if t:
        texts.append(t)
del ds

print(f"Loaded {len(texts)} text chunks")

all_ids = []
for t in texts:
    ids = tokenizer.encode(t, add_special_tokens=False)
    if len(ids) > 0:
        all_ids.extend(ids)
del texts

data = torch.tensor(all_ids, dtype=torch.long)
del all_ids
n = (len(data) - 1) // SEQ
data = data[:n * SEQ + 1]
train_tokens = data.shape[0]
print(f"Training tokens: {train_tokens}")

v3 = GenoTransformer(vocab_size=VOCAB, hidden_dim=D, num_layers=L, num_heads=HEADS, rank=RANK).to(device)
tf = BaselineTransformer(vocab_size=VOCAB, hidden_dim=D, num_layers=L, num_heads=HEADS).to(device)

v3_params = sum(p.numel() for p in v3.parameters() if p.requires_grad)
tf_params = sum(p.numel() for p in tf.parameters() if p.requires_grad)
v3_eff = L * 12 * D * D
print(f"V3 stored params: {v3_params/1e6:.1f}M, effective: {v3_eff/1e6:.1f}M")
print(f"TF stored params: {tf_params/1e6:.1f}M")

v3_scaler = torch.amp.GradScaler("cuda")
tf_scaler = torch.amp.GradScaler("cuda")
v3_opt = torch.optim.AdamW(v3.parameters(), lr=3e-4, weight_decay=0.01)
tf_opt = torch.optim.AdamW(tf.parameters(), lr=3e-4, weight_decay=0.01)

def get_batch(data, batch_size, seq_len):
    n = len(data) - seq_len - 1
    idx = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x.to(device), y.to(device)

def evaluate(model, data, batch_size, seq_len, num_batches=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, batch_size, seq_len)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens

print(f"\n{'Step':>5} | {'V3 Loss':>8} | {'TF Loss':>8} | {'V3 PPL':>8} | {'TF PPL':>8} | {'VRAM':>6}")
print("-" * 60)

t_start = time.time()
for step in range(1, STEPS + 1):
    x, y = get_batch(data, B, SEQ)

    v3_opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        l_v3 = F.cross_entropy(v3(x).reshape(-1, VOCAB), y.reshape(-1))
    v3_scaler.scale(l_v3).backward()
    v3_scaler.unscale_(v3_opt)
    torch.nn.utils.clip_grad_norm_(v3.parameters(), 1.0)
    v3_scaler.step(v3_opt)
    v3_scaler.update()

    tf_opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        l_tf = F.cross_entropy(tf(x).reshape(-1, VOCAB), y.reshape(-1))
    tf_scaler.scale(l_tf).backward()
    tf_scaler.unscale_(tf_opt)
    torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
    tf_scaler.step(tf_opt)
    tf_scaler.update()

    if step % 500 == 0 or step == 1:
        v3_ppl = math.exp(evaluate(v3, data, B, SEQ))
        tf_ppl = math.exp(evaluate(tf, data, B, SEQ))
        vram = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"{step:5d} | {l_v3.item():8.4f} | {l_tf.item():8.4f} | {v3_ppl:8.2f} | {tf_ppl:8.2f} | {vram:5.0f}MB")
        torch.cuda.reset_peak_memory_stats(device)

t_total = time.time() - t_start
print("-" * 60)
print(f"Total time: {t_total:.0f}s ({t_total/STEPS*1000:.0f}ms/step)")

print(f"\nFinal:")
v3_ppl = math.exp(evaluate(v3, data, B, SEQ, num_batches=200))
tf_ppl = math.exp(evaluate(tf, data, B, SEQ, num_batches=200))
print(f"V3 perplexity: {v3_ppl:.2f}")
print(f"TF perplexity: {tf_ppl:.2f}")
v3_win = (tf_ppl - v3_ppl) / tf_ppl * 100
print(f"V3 improvement: {v3_win:.1f}% better perplexity" if v3_win > 0 else f"TF is {abs(v3_win):.1f}% better")
print(f"V3 params: {v3_params/1e6:.1f}M (effective: {v3_eff/1e6:.1f}M)")
print(f"TF params:  {tf_params/1e6:.1f}M")
print(f"Compression: {tf_params/v3_params:.1f}x fewer stored params for V3")
