import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from condenser.adaptive import AdaptivePRISM, AdaptivePRISMLoss

model = AdaptivePRISM(vocab_size=500, hidden_dim=128, num_layers=4, rank=8, state_dim=64, budget_reg=0.01)
loss_fn = AdaptivePRISMLoss(budget_reg=0.01)

x = torch.randint(0, 500, (2, 16))
logits, budget_penalty = model(x)
loss, details = loss_fn(logits, x, budget_penalty)
loss.backward()

print(f"PASS Forward+backward: loss={loss.item():.4f}")
print(f"PASS CE={details['ce']:.4f} budget_penalty={details['budget']:.4f}")
print(f"PASS Trained params: {model.count_trained():,}")
print(f"PASS Budget penalty: {budget_penalty.item():.4f}")
