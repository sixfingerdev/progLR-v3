"""
V3-Embed + Shared Transformer Layer (ALBERT-style).
1 shared TF layer loops L_virtual times. V3-Embed for 500× embedding compression.
Tests: does weight sharing preserve OOD reasoning while compressing 10×?
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from condenser.v3_embed import V3WeightEmbedding
from condenser.baseline_transformer import BaselineAttention


class SharedFFN(nn.Module):
    """Standard full-rank FFN (shared across layers)."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.W1 = nn.Linear(dim, dim*expansion, bias=False)
        self.W2 = nn.Linear(dim*expansion, dim, bias=False)
    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


class SharedTFBlock(nn.Module):
    """Single transformer block: attention + FFN + norms."""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = BaselineAttention(dim, heads)
        self.ffn = SharedFFN(dim, 4)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class V3SharedTF(nn.Module):
    """V3-Embed + 1 shared TF block applied L times."""
    def __init__(self, vocab_size, hidden_dim, num_virtual_layers, num_heads, embed_rank=8, alpha=64):
        super().__init__()
        self.num_virtual_layers = num_virtual_layers
        self.v3_embed = V3WeightEmbedding(vocab_size, hidden_dim, embed_rank, alpha)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.shared_block = SharedTFBlock(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.v3_embed.forward_embed(input_ids) + self.pos_embedding[:, :T]
        for _ in range(self.num_virtual_layers):
            x = self.shared_block(x)
        return self.v3_embed.forward_head(self.norm(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Standard (non-shared) baseline
class StandardFFN(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.W1 = nn.Linear(dim, dim*expansion, bias=False)
        self.W2 = nn.Linear(dim*expansion, dim, bias=False)
    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))

class StandardBlock(nn.Module):
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

class V3StandardTF(nn.Module):
    """V3-Embed + L separate TF layers (baseline)."""
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, embed_rank=8, alpha=64):
        super().__init__()
        self.v3_embed = V3WeightEmbedding(vocab_size, hidden_dim, embed_rank, alpha)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([StandardBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.v3_embed.forward_embed(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers: x = layer(x)
        return self.v3_embed.forward_head(self.norm(x))
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_ood_data(NUM=100, device='cuda'):
    PLUS, EQUALS, VOCAB = NUM, NUM+1, NUM+2
    train = torch.tensor([[a,PLUS,b,EQUALS,(a+b)%NUM] for a in range(50) for b in range(NUM)], dtype=torch.long, device=device)
    test  = torch.tensor([[a,PLUS,b,EQUALS,(a+b)%NUM] for a in range(50,NUM) for b in range(NUM)], dtype=torch.long, device=device)
    return VOCAB, train, test

def accuracy(model, data, B=256):
    model.eval(); c=0; t=0
    with torch.no_grad():
        for i in range(0, data.shape[0], B):
            b = data[i:i+B]; preds = model(b[:,:4]).argmax(dim=-1)
            c += (preds[:,-1]==b[:,-1]).sum().item(); t+=b.shape[0]
    return c/t*100


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n', flush=True)

    D=192; HEADS=6; RANK=16; ALPHA=64; B=64; STEPS=10000
    VOCAB, train_data, test_data = make_ood_data(100, device)
    print(f'Train: {train_data.shape[0]}, Test: {test_data.shape[0]}, Vocab: {VOCAB}\n', flush=True)

    configs = [
        # Standard (non-shared) — each layer has its own params
        ('Std-1L', V3StandardTF(VOCAB, D, 1, HEADS, RANK, ALPHA)),
        ('Std-2L', V3StandardTF(VOCAB, D, 2, HEADS, RANK, ALPHA)),
        ('Std-4L', V3StandardTF(VOCAB, D, 4, HEADS, RANK, ALPHA)),
        # Shared — 1 block applied L times
        ('Shr-x2', V3SharedTF(VOCAB, D, 2, HEADS, RANK, ALPHA)),
        ('Shr-x4', V3SharedTF(VOCAB, D, 4, HEADS, RANK, ALPHA)),
        ('Shr-x8', V3SharedTF(VOCAB, D, 8, HEADS, RANK, ALPHA)),
        ('Shr-x12', V3SharedTF(VOCAB, D, 12, HEADS, RANK, ALPHA)),
    ]

    print(f'{"Model":<12} {"Params":>10} {"Comp vs Std-2L":>16}')
    print(f'{"-"*40}')
    std2_params = configs[1][1].count_params()
    for n,m in configs:
        p = m.count_params(); comp = std2_params / p
        print(f'  {n:<10} {p:>10,} {comp:>14.1f}×', flush=True)

    results = {}
    for name, model in configs:
        from condenser.genomatrix import _R_POOL; _R_POOL.clear()
        torch.cuda.reset_peak_memory_stats(device)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        best_ood = 0
        print(f'\n{"="*55}', flush=True)
        print(f'  {name}: {model.count_params():,} params, {model.num_virtual_layers if hasattr(model,"num_virtual_layers") else len(model.layers)} virtual layers', flush=True)
        print(f'{"="*55}', flush=True)
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
                print(f'  step={step:>5} | train={train_acc:>5.1f}% | ood={ood_acc:>5.1f}%', flush=True)
        train_acc = accuracy(model, train_data)
        ood_acc = accuracy(model, test_data)
        best_ood = max(best_ood, ood_acc)
        results[name] = {'train': train_acc, 'ood': ood_acc, 'best_ood': best_ood, 'params': model.count_params()}
        print(f'  FINAL | train={train_acc:>5.1f}% | ood={ood_acc:>5.1f}% | best_ood={best_ood:>5.1f}%', flush=True)

    print(f'\n{"="*70}')
    print(f'  FINAL: V3-Embed + Shared TF vs Standard TF')
    print(f'{"="*70}')
    print(f'  {"Model":<10} {"Params":>8} {"VirtL":>5} {"Train":>7} {"OOD":>7} {"BestOOD":>8} {"Comp":>8}')
    print(f'  {"-"*58}')
    for name, r in sorted(results.items()):
        comp = std2_params / r['params']
        virt = name.split('-')[1].replace('x','').replace('L','') if 'Shr' in name else name.split('-')[1].replace('L','')
        print(f'  {name:<10} {r["params"]:>8,} {virt:>5} {r["train"]:>6.1f}% {r["ood"]:>6.1f}% {r["best_ood"]:>7.1f}% {comp:>7.1f}×')
