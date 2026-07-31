import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CondenserLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        atom_dim: int = 64,
        n_atoms_in_pool: int = 10000,
        n_selected: int = 16,
        genom_hidden: int = 128,
        tau: float = 0.5,
        pool_seed: int = 42,
        layer_id: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_dim = atom_dim
        self.n_atoms_in_pool = n_atoms_in_pool
        self.n_selected = n_selected
        self.tau = tau
        self.layer_id = layer_id

        rng = torch.Generator()
        rng.manual_seed(pool_seed + layer_id)
        pool = torch.randn(n_atoms_in_pool, atom_dim, generator=rng)
        pool = pool / pool.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer('atom_pool', pool)

        self.genom = nn.Sequential(
            nn.Linear(hidden_dim + 1, genom_hidden),
            nn.ReLU(),
            nn.Linear(genom_hidden, atom_dim),
        )

        rng_exp = torch.Generator()
        rng_exp.manual_seed(pool_seed + layer_id + 1000)
        W_down = torch.randn(hidden_dim, atom_dim, generator=rng_exp) / math.sqrt(hidden_dim)
        W_up = torch.randn(atom_dim, hidden_dim, generator=rng_exp) / math.sqrt(atom_dim)
        self.register_buffer('W_down', W_down)
        self.register_buffer('W_up', W_up)

        rng_bias = torch.Generator()
        rng_bias.manual_seed(pool_seed + layer_id + 2000)
        self.register_buffer('bias_fixed', torch.randn(hidden_dim, generator=rng_bias) * 0.01)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device

        residual = x

        x_norm = self.norm(x)

        layer_embed = torch.full((B, 1), self.layer_id, device=device).expand(B, 1)
        genom_in = torch.cat([x_norm.mean(dim=1), layer_embed], dim=-1)
        code = self.genom(genom_in)

        pool = self.atom_pool.to(device)
        scores = code @ pool.T
        atom_weights = F.gumbel_softmax(scores, tau=self.tau, hard=True, dim=-1)

        combined_atom = atom_weights @ pool

        x_proj = x_norm @ self.W_down
        gated = x_proj * combined_atom.unsqueeze(1)
        out = gated @ self.W_up + self.bias_fixed

        out = F.relu(out)

        return out + residual


class CondenserModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 128,
        num_layers: int = 6,
        atom_dim: int = 64,
        n_atoms_in_pool: int = 10000,
        n_selected: int = 16,
        genom_hidden: int = 128,
        tau: float = 0.5,
        pool_seed: int = 42,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)

        self.layers = nn.ModuleList([
            CondenserLayer(
                hidden_dim=hidden_dim,
                atom_dim=atom_dim,
                n_atoms_in_pool=n_atoms_in_pool,
                n_selected=n_selected,
                genom_hidden=genom_hidden,
                tau=tau,
                pool_seed=pool_seed + i,
                layer_id=i,
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :T, :]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_buffers(self):
        return sum(b.numel() for b in self.buffers())

    def memory_usage_mb(self):
        total_bytes = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        buffer_bytes = sum(
            b.numel() * b.element_size() for b in self.buffers()
        )
        return {
            'trainable_params_mb': total_bytes / (1024 * 1024),
            'buffers_mb': buffer_bytes / (1024 * 1024),
        }
