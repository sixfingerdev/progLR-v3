"""
PRISM: Parametric Random Instance-Specific Modulation v2

More efficient version using low-rank random projections.
Instead of generating full hidden×hidden matrices, generates
random rank-1 factors that are combined by the Genom modulation.

W_eff = sum_i mod_i * (u_i @ v_i^T) = U @ diag(mod) @ V^T

Memory: only Genom params are stored (~50-100M)
Random basis: regenerated from deterministic seeds each forward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SeedRNG:
    @staticmethod
    def randn(seed: int, shape: tuple, device: torch.device, dtype: torch.dtype = torch.float32):
        gen = torch.Generator(device=device)
        gen.manual_seed(abs(seed) % (2**31 - 1))
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype)


class PRISMLayer(nn.Module):
    """
    A single layer with ephemeral weight generation.
    W_eff = U @ diag(modulation) @ V^T
    where U, V are random matrices from deterministic seeds.
    """
    def __init__(self, hidden_dim: int, rank: int = 8, seed: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.seed_u = seed * 2 + 1000
        self.seed_v = seed * 2 + 1001

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        B, T, D = x.shape
        k = modulation.shape[-1]

        U = SeedRNG.randn(self.seed_u, (D, k), x.device, x.dtype)
        V = SeedRNG.randn(self.seed_v, (D, k), x.device, x.dtype)

        U = U / math.sqrt(k)
        V = V / math.sqrt(k)

        tmp = x @ V
        scaled = tmp * modulation.unsqueeze(1)
        out = scaled @ U.T

        out = F.relu(out)
        return out + residual


class GenomNet(nn.Module):
    def __init__(self, state_dim: int = 256, output_dim: int = 8, n_layers_embed: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.layer_embed = nn.Embedding(n_layers_embed + 1, state_dim)
        self.state_init = nn.Parameter(torch.randn(1, state_dim) * 0.02)

        self.cell = nn.GRUCell(state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, output_dim)

    def forward(self, n_layers: int, device: torch.device) -> list[torch.Tensor]:
        h = self.state_init.expand(1, -1).to(device)
        modulations = []

        for l in range(n_layers):
            l_emb = self.layer_embed(torch.tensor([l], device=device))
            h = self.cell(l_emb, h)
            mod = self.out_proj(h)
            modulations.append(mod)

        return modulations


class PRISMModel(nn.Module):
    """
    Full language model with PRISM ephemeral weight generation.
    Only Genom + embeddings are trained.
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 4096,
        num_layers: int = 64,
        state_dim: int = 256,
        rank: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.genom = GenomNet(
            state_dim=state_dim,
            output_dim=rank,
            n_layers_embed=num_layers,
        )

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            PRISMLayer(hidden_dim, rank, seed=i * 100)
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)

        modulations = self.genom(self.num_layers, x.device)

        for i, layer in enumerate(self.layers):
            mod = modulations[i]
            B = x.shape[0]
            if mod.shape[0] < B:
                mod = mod.expand(B, -1)
            x = layer(x, mod)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def count_trained(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def effective_params(self):
        per_layer = self.hidden_dim * self.hidden_dim * 2
        return per_layer * self.num_layers

    def memory_report(self):
        trained = self.count_trained()
        eff = self.effective_params()

        model_gb = trained * 2 / 1e9
        optim_gb = trained * 8 / 1e9
        act_gb = self.hidden_dim * 1024 * 4 / 1e9

        return {
            'trained_m': trained / 1e6,
            'effective_b': eff / 1e9,
            'compression': eff / trained if trained else 0,
            'model_gb': model_gb,
            'optim_gb': optim_gb,
            'act_gb': act_gb,
            'total_gb': model_gb + optim_gb + act_gb,
            'fits_6gb': (model_gb + optim_gb + act_gb) < 6.0,
        }
