"""
WikiText-103 BPE Benchmark: V3 vs Transformer with VALIDATION split
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
SEQ = 128; B = 8; D = 768; L = 12; HEADS = 12; RANK = 6; STEPS = 4000

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
VOCAB = tokenizer.vocab_size

print(f"V3 (D={D}, L={L}, rank={RANK}) vs TF (D={D}, L={L})")
print(f"WikiText-103 BPE (vocab={VOCAB}) Batch={B} Seq={SEQ} Steps={STEPS}")
print("Loading data...", flush=True)

def load_and_tokenize(split, n_max=None):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        if n_max and i >= n_max: break
        t = ex["text"].strip()
        if t: texts.append(t)
    all_ids = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids): all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)

train_data = load_and_tokenize("train", 200000)
nt = (len(train_data)-1)//SEQ
train_data = train_data[:nt*SEQ+1]
print(f"Train tokens: {train_data.shape[0]}")

val_data = load_and_tokenize("validation")
nv = (len(val_data)-1)//SEQ
val_data = val_data[:nv*SEQ+1]
print(f"Val tokens: {val_data.shape[0]}")

v3 = GenoTransformer(vocab_size=VOCAB, hidden_dim=D, num_layers=L, num_heads=HEADS, rank=RANK).to(device)
tf = BaselineTransformer(vocab_size=VOCAB, hidden_dim=D, num_layers=L, num_heads=HEADS).to(device)

v3_p = sum(p.numel() for p in v3.parameters() if p.requires_grad)
tf_p = sum(p.numel() for p in tf.parameters() if p.requires_grad)
print(f"V3: {v3_p/1e6:.1f}M stored, {L*12*D*D/1e6:.1f}M effective")
print(f"TF: {tf_p/1e6:.1f}M stored")

v3_scaler = torch.amp.GradScaler("cuda")
tf_scaler = torch.amp.GradScaler("cuda")
v3_opt = torch.optim.AdamW(v3.parameters(), lr=3e-4, weight_decay=0.01)
tf_opt = torch.optim.AdamW(tf.parameters(), lr=3e-4, weight_decay=0.01)

def get_batch(d, bs, sl):
    idx = torch.randint(0, len(d)-sl-1, (bs,))
    x = torch.stack([d[i:i+sl] for i in idx]).to(device)
    y = torch.stack([d[i+1:i+sl+1] for i in idx]).to(device)
    return x, y

def val_ppl(model, data, bs, sl, num_batches=100):
    model.eval()
    total_loss = 0.0; total = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, bs, sl)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
            total_loss += loss.item()
            total += y.numel()
    model.train()
    return math.exp(total_loss / total)

print(f"\n{'Step':>5} | {'V3 Trn':>8} | {'TF Trn':>8} | {'V3 Val':>8} | {'TF Val':>8} | {'VRAM':>6}")
print("-" * 60)

t_start = time.time()
for step in range(1, STEPS + 1):
    x, y = get_batch(train_data, B, SEQ)

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
        v3_v = val_ppl(v3, val_data, B, SEQ)
        tf_v = val_ppl(tf, val_data, B, SEQ)
        vram = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"{step:5d} | {l_v3.item():8.4f} | {l_tf.item():8.4f} | {v3_v:8.2f} | {tf_v:8.2f} | {vram:5.0f}MB")
        torch.cuda.reset_peak_memory_stats(device)

t_total = time.time() - t_start
print("-" * 60)
print(f"Time: {t_total:.0f}s ({t_total/STEPS*1000:.0f}ms/step)")

print(f"\nFinal Validation:")
v3_v = val_ppl(v3, val_data, B, SEQ, 500)
tf_v = val_ppl(tf, val_data, B, SEQ, 500)
print(f"V3 valid PPL: {v3_v:.2f}")
print(f"TF valid PPL:  {tf_v:.2f}")
print(f"V3 stored params: {v3_p/1e6:.1f}M")
print(f"TF stored params:  {tf_p/1e6:.1f}M")
