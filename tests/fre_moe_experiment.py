"""
FRE-MoE: Frozen Random Expert Mixture of Experts.
FFN replaced with K frozen random (W1,W2) experts + learned gating.
"""
import sys, os, math, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from condenser.v3_embed import V3WeightEmbedding
from condenser.baseline_transformer import BaselineAttention


class FREMoEFFN(nn.Module):
    """K frozen random (W1,W2) pairs + learned gating, no learned correction."""
    def __init__(self, dim, expansion=4, K=8, seed=42, layer_id=0):
        super().__init__()
        self.K = K
        hidden_dim = dim * expansion
        gen = torch.Generator().manual_seed(seed + layer_id * 1001)
        W1 = torch.randn(K, hidden_dim, dim, generator=gen) / math.sqrt(dim)
        W2 = torch.randn(K, dim, hidden_dim, generator=gen) / math.sqrt(hidden_dim)
        self.register_buffer('W1', W1)
        self.register_buffer('W2', W2)
        self.gate = nn.Sequential(
            nn.Linear(dim, max(dim // 4, 16)), nn.ReLU(),
            nn.Linear(max(dim // 4, 16), K),
        )

    def forward(self, x):
        B, T, D = x.shape
        g = F.softmax(self.gate(x), dim=-1)
        out = x.new_zeros(B, T, D)
        for k in range(self.K):
            h = F.relu(x @ self.W1[k].T)
            out += g[:, :, k:k+1] * (h @ self.W2[k].T)
        return out


class FREMoELayer(nn.Module):
    def __init__(self, dim, heads, K=8, seed=42, layer_id=0):
        super().__init__()
        self.attn = BaselineAttention(dim, heads)
        self.ffn = FREMoEFFN(dim, 4, K, seed, layer_id)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FREMoETransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, K=8, embed_rank=8, alpha=64, seed=42):
        super().__init__()
        self.v3_embed = V3WeightEmbedding(vocab_size, hidden_dim, embed_rank, alpha)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([
            FREMoELayer(hidden_dim, num_heads, K, seed, i) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.v3_embed.forward_embed(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers: x = layer(x)
        return self.v3_embed.forward_head(self.norm(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StandardFFN(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.W1 = nn.Linear(dim, dim*expansion, bias=False)
        self.W2 = nn.Linear(dim*expansion, dim, bias=False)
    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))

class StandardLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = BaselineAttention(dim, heads)
        self.ffn = StandardFFN(dim, 4)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class V3EmbedBaseline(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, embed_rank=8, alpha=64):
        super().__init__()
        self.v3_embed = V3WeightEmbedding(vocab_size, hidden_dim, embed_rank, alpha)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([StandardLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.v3_embed.forward_embed(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers: x = layer(x)
        return self.v3_embed.forward_head(self.norm(x))
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Data ---
def make_ood_data(NUM=100, device='cuda'):
    PLUS, EQUALS, VOCAB = NUM, NUM + 1, NUM + 2
    train = torch.tensor([[a, PLUS, b, EQUALS, (a+b)%NUM] for a in range(50) for b in range(NUM)], dtype=torch.long, device=device)
    test = torch.tensor([[a, PLUS, b, EQUALS, (a+b)%NUM] for a in range(50, NUM) for b in range(NUM)], dtype=torch.long, device=device)
    return VOCAB, train, test

def accuracy(model, data, B=256):
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]
            preds = model(b[:,:4]).argmax(dim=-1)
            correct += (preds[:,-1]==b[:,-1]).sum().item(); total+=b.shape[0]
    return correct/total*100


# --- Main ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n', flush=True)

    D=192; L=2; HEADS=6; RANK=16; ALPHA=64; B=64; STEPS=10000
    VOCAB, train_data, test_data = make_ood_data(100, device)
    print(f'Train: {train_data.shape[0]}, Test: {test_data.shape[0]}, Vocab: {VOCAB}', flush=True)

    configs = [
        ('1-Baseline TF', V3EmbedBaseline(VOCAB, D, L, HEADS, RANK, ALPHA)),
        ('2-FRE-MoE K=4', FREMoETransformer(VOCAB, D, L, HEADS, K=4, embed_rank=RANK, alpha=ALPHA)),
        ('3-FRE-MoE K=8', FREMoETransformer(VOCAB, D, L, HEADS, K=8, embed_rank=RANK, alpha=ALPHA)),
        ('4-FRE-MoE K=16', FREMoETransformer(VOCAB, D, L, HEADS, K=16, embed_rank=RANK, alpha=ALPHA)),
        ('5-FRE-MoE K=32', FREMoETransformer(VOCAB, D, L, HEADS, K=32, embed_rank=RANK, alpha=ALPHA)),
    ]

    print(f'\n{"Model":<20} {"Params":>10}', flush=True)
    for n,m in configs:
        print(f'  {n:<18} {m.count_params():>10,}', flush=True)

    results = {}
    for name, model in configs:
        from condenser.genomatrix import _R_POOL; _R_POOL.clear()
        torch.cuda.reset_peak_memory_stats(device)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        best_ood=0
        print(f'\n{"="*60}', flush=True)
        print(f'  TRAINING {name} ({model.count_params():,} params)', flush=True)
        print(f'{"="*60}', flush=True)
        for step in range(STEPS):
            model.train()
            idx = torch.randint(0, train_data.shape[0], (B,))
            batch = train_data[idx]
            loss = F.cross_entropy(model(batch[:,:4]).reshape(-1,VOCAB), batch[:,1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % 5000 == 0:
                train_acc = accuracy(model, train_data)
                ood_acc = accuracy(model, test_data)
                best_ood = max(best_ood, ood_acc)
                vram = torch.cuda.memory_allocated(device)/1e6
                print(f'  step={step:>5} | train={train_acc:>5.1f}% | ood={ood_acc:>5.1f}% | vram={vram:.0f}MB', flush=True)
        train_acc = accuracy(model, train_data)
        ood_acc = accuracy(model, test_data)
        best_ood = max(best_ood, ood_acc)
        results[name] = {'train': train_acc, 'ood': ood_acc, 'best_ood': best_ood, 'params': model.count_params()}
        print(f'  FINAL | train={train_acc:>5.1f}% | ood={ood_acc:>5.1f}% | best_ood={best_ood:>5.1f}%', flush=True)

    print(f'\n{"="*60}')
    print(f'  FINAL RESULTS')
    print(f'{"="*60}')
    print(f'  {"Model":<20} {"Params":>8} {"Train":>7} {"OOD":>7} {"BestOOD":>8}')
    print(f'  {"-"*55}')
    for name, r in sorted(results.items()):
        print(f'  {name:<20} {r["params"]:>8,} {r["train"]:>6.1f}% {r["ood"]:>6.1f}% {r["best_ood"]:>7.1f}%')
