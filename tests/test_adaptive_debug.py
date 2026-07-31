import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from condenser.adaptive import AdaptivePRISM, AdaptivePRISMLoss

model = AdaptivePRISM(vocab_size=500, hidden_dim=128, num_layers=2, rank=4, state_dim=64, budget_reg=0.01)

x = torch.randint(0, 500, (2, 8))

with torch.no_grad():
    embed = model.token_embedding(x)
    print(f"Embedded shape: {embed.shape}")

    mods, buds, total = model.genom(embed, 2, torch.ones(2, 8, dtype=torch.bool, device=embed.device))
    print(f"Num modulations: {len(mods)}, first mod shape: {mods[0].shape}")
    print(f"Num budgets: {len(buds)}, first budget shape: {buds[0].shape}")

    h = embed
    for l in range(2):
        print(f"Layer {l}: input shape {h.shape}")
        h = model.layers[l](h, mods[l], buds[l], torch.ones(2, 8, dtype=torch.bool, device=h.device))
        print(f"Layer {l}: output shape {h.shape}")

print("All shapes OK!")
