"""
Adaptive Compute Budget for PRISM.
Each token gets a learned budget allocation from Genom.
Simple tokens skip layers, complex tokens get full compute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptivePRISMLayer(nn.Module):
    """
    PRISM layer with adaptive compute budget.
    Genom produces: modulation (k-dim) + budget (1-dim per token)
    """
    def __init__(self, hidden_dim: int, rank: int = 8, seed: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.seed_u = seed * 2 + 100
        self.seed_v = seed * 2 + 101

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, modulation: torch.Tensor,
                budget: torch.Tensor, active_mask: torch.Tensor) -> tuple:
        residual = x

        B, T, D = x.shape

        U = torch.randn(D, self.rank, device=x.device) / math.sqrt(self.rank)
        V = torch.randn(D, self.rank, device=x.device) / math.sqrt(self.rank)

        tmp = x @ V
        scaled = tmp * modulation
        out = scaled @ U.T

        out = F.relu(out)

        out = torch.where(active_mask.unsqueeze(-1), out, residual)

        return out


class AdaptiveGenom(nn.Module):
    """
    Genom that produces modulation + budget per layer per token.
    """
    def __init__(self, hidden_dim: int, rank: int, state_dim: int = 256, n_layers: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.state_dim = state_dim

        self.token_proj = nn.Linear(hidden_dim, state_dim)
        self.layer_embed = nn.Embedding(n_layers + 1, state_dim)

        self.state_gru = nn.GRUCell(state_dim * 2, state_dim)
        self.state_init = nn.Parameter(torch.randn(1, state_dim) * 0.01)

        self.mod_head = nn.Linear(state_dim, rank)
        self.budget_head = nn.Linear(state_dim, 1)

    def forward(self, x: torch.Tensor, n_layers: int, active_mask: torch.Tensor):
        B, T, D = x.shape
        device = x.device

        token_state = self.token_proj(x)
        h = self.state_init.expand(B * T, -1).to(device)

        modulations = []
        budgets = []
        total_budget = x.new_zeros(B, T)

        for l in range(n_layers):
            l_emb = self.layer_embed(torch.tensor([l], device=device)).expand(B * T, -1)
            h = self.state_gru(torch.cat([token_state.reshape(-1, self.state_dim), l_emb], dim=-1), h)

            mod = self.mod_head(h).view(B, T, self.rank)
            bud = torch.sigmoid(self.budget_head(h)).view(B, T)

            bud = bud * active_mask

            total_budget = total_budget + bud

            modulations.append(mod)
            budgets.append(bud)

        return modulations, budgets, total_budget


class AdaptivePRISM(nn.Module):
    """
    Full model with adaptive compute budget.
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        rank: int = 16,
        state_dim: int = 256,
        budget_reg: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.budget_reg = budget_reg

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.genom = AdaptiveGenom(
            hidden_dim=hidden_dim,
            rank=rank,
            state_dim=state_dim,
            n_layers=num_layers,
        )

        self.layers = nn.ModuleList([
            AdaptivePRISMLayer(hidden_dim, rank, seed=i)
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        self.early_classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> tuple:
        B, T = input_ids.shape
        x = self.token_embedding(input_ids)

        active_mask = x.new_ones(B, T, dtype=torch.bool)

        modulations, budgets, total_budget = self.genom(x, self.num_layers, active_mask)

        total_compute = x.new_zeros(B, T)

        for l in range(self.num_layers):
            target_compute = (l + 1) / self.num_layers

            layer_active = active_mask

            x = self.layers[l](x, modulations[l], budgets[l], layer_active)

            used = budgets[l] * layer_active.float()
            total_compute = total_compute + used

        x = self.final_norm(x)
        logits = self.lm_head(x)

        budget_penalty = total_compute.mean()

        return logits, budget_penalty

    def count_trained(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptivePRISMLoss(nn.Module):
    def __init__(self, budget_reg: float = 0.01):
        super().__init__()
        self.budget_reg = budget_reg
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets, budget_penalty):
        ce_loss = self.ce(logits[:, :-1].reshape(-1, logits.shape[-1]), targets[:, 1:].reshape(-1))
        total = ce_loss + self.budget_reg * budget_penalty
        return total, {'ce': ce_loss.item(), 'budget': budget_penalty.item()}
