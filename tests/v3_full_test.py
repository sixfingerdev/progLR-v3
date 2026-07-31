"""
Build and test V3-Full: V3 weight generation for ALL layers.
- Embedding/Head: V3WeightEmbedding (R + LoRA)
- Transformer layers: GenoTransformerLayer (R + LoRA)
"""
import sys, os, math, random, time
sys.path.insert(0, 'C:\\Users\\enesa\\law')
import torch
import torch.nn as nn
import torch.nn.functional as F

from condenser.v3_embed import V3WeightEmbedding, V3EmbedTransformer
from condenser.genomatrix import GenoTransformerLayer, _R_POOL, GenoTransformer
from condenser.baseline_transformer import BaselineTransformer


class V3FullTransformer(nn.Module):
    """End-to-end V3 compression: embedding + all layers + head."""
    def __init__(self, vocab_size=50000, hidden_dim=768, num_layers=12,
                 num_heads=12, rank=8, alpha=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        self.v3_embed = V3WeightEmbedding(vocab_size, hidden_dim, rank, alpha)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)

        self.layers = nn.ModuleList([
            GenoTransformerLayer(hidden_dim, num_heads, rank, seed=42, layer_id=i, alpha=alpha)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Head shares V3 embedding weights (tied)
        # V3WeightEmbedding.forward_head already handles the LM head

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.v3_embed.forward_embed(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=True)
        x = self.norm(x)
        return self.v3_embed.forward_head(x)

    def count_trained(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self):
        return sum(p.numel() for p in self.parameters())


# Quick arithmetic test
device = torch.device('cuda')
NUM = 100; PLUS = NUM; EQUALS = NUM + 1; VOCAB = NUM + 2
random.seed(42)
pairs = [(a, b) for a in range(NUM) for b in range(NUM)]
c_sum = [(a + b) % NUM for a, b in pairs]
indices = list(range(len(pairs))); random.shuffle(indices)
split = int(len(indices) * 0.8)
def make_data(idx_list):
    s = []
    for idx in idx_list:
        a, b = pairs[idx]; c = c_sum[idx]
        s.append([a, PLUS, b, EQUALS, c])
    return torch.tensor(s, dtype=torch.long, device=device)
td = make_data(indices[:split]); vd = make_data(indices[split:])
B = 64; STEPS = 3000; D, H, L = 192, 6, 12

def acc(model, data):
    model.eval(); c=0; t=0
    with torch.no_grad():
        for i in range(0,data.shape[0],B):
            b = data[i:i+B]; preds = model(b[:,:4]).argmax(dim=-1)
            c += (preds[:,-1]==b[:,-1]).sum().item(); t+=b.shape[0]
    return c/t*100

def run(name, model_fn):
    print(f'\n=== {name} ===', flush=True)
    _R_POOL.clear()
    torch.manual_seed(42)
    model = model_fn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best = 0; start = time.time()
    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, td.shape[0], (B,)); batch = td[idx]
        loss = F.cross_entropy(model(batch[:,:4]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            ac = acc(model, vd); best = max(best, ac)
    ac = acc(model, vd); best = max(best, ac)
    p_trained = model.count_trained()
    p_total = model.count_total() if hasattr(model, 'count_total') else p_trained
    tf_params = sum(p.numel() for p in BaselineTransformer(VOCAB, D, L, H).parameters() if p.requires_grad)
    comp = tf_params / max(p_trained, 1)
    print(f'  {name}: trained {p_trained:,} total {p_total:,} acc {ac:.1f}% best {best:.1f}% comp {comp:.1f}x [{time.time()-start:.0f}s]', flush=True)
    return p_trained, p_total, ac, comp

# V3-Full at various ranks
for rank in [2, 4, 8]:
    def make(rank=rank):
        return V3FullTransformer(VOCAB, D, L, H, rank=rank)
    run(f'V3-Full r={rank}', make)

# Reference: GenoTransformer (V3 layers only, standard embed) 
def make_v3():
    return GenoTransformer(VOCAB, D, L, L, H, 1.0)
run('GenoTransformer (old V3)', make_v3)

# Reference: Standard TF
def make_tf():
    return BaselineTransformer(VOCAB, D, L, H)
run('Standard TF', make_tf)

print(f'\n{"="*60}', flush=True)
print('Compression analysis at GPT-2 XL scale (D=1600, L=48, vocab=50K):', flush=True)
total_std = 1558_000_000  # 1.5B
embed_std = 50000 * 1600  # 80M
layers_std = total_std - embed_std  # 1.478B
for r in [2, 4, 8]:
    embed_v3 = 50000 * r + 1600 * r
    layers_v3_v3 = 48 * 12 * r * 1600 * 2  # approx: 12 matrices × 2D×r per layer
    total_v3_full = embed_v3 + layers_v3_v3
    print(f'  r={r}: embed {embed_v3/1e6:.1f}M ({embed_std/embed_v3:.0f}x) layers {layers_v3_v3/1e6:.1f}M total {total_v3_full/1e6:.1f}M = {total_std/total_v3_full:.1f}x')
