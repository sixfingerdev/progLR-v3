import torch
import torch.nn as nn
import math

def create_fixed_random(dim_in, dim_out, seed, device='cpu'):
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    M = torch.randn(dim_out, dim_in, generator=rng, device=device)
    M = M / math.sqrt(dim_in)
    return M

class Rank1Synthesis(nn.Module):
    def __init__(self, atom_dim, hidden_dim, n_components=16, seed=999):
        super().__init__()
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.n_components = n_components

        left_seeds = [seed + i for i in range(n_components)]
        right_seeds = [seed + n_components + i for i in range(n_components)]

        left = torch.stack([
            create_fixed_random(atom_dim, hidden_dim, s) for s in left_seeds
        ])
        right = torch.stack([
            create_fixed_random(atom_dim, hidden_dim, s) for s in right_seeds
        ])

        self.register_buffer('left_basis', left)
        self.register_buffer('right_basis', right)

    def forward(self, combined_atom):
        B = combined_atom.shape[0]

        left_vectors = (combined_atom.unsqueeze(1) * self.left_basis.unsqueeze(0)).sum(dim=-1)
        right_vectors = (combined_atom.unsqueeze(1) * self.right_basis.unsqueeze(0)).sum(dim=-1)

        W = left_vectors.unsqueeze(-1) * right_vectors.unsqueeze(-2)
        W = W.sum(dim=1)
        return W


class GatedSynthesis(nn.Module):
    def __init__(self, atom_dim, hidden_dim, seed=999):
        super().__init__()
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim

        self.register_buffer('W_proj', create_fixed_random(atom_dim, hidden_dim, seed))
        self.register_buffer('W_gate', create_fixed_random(atom_dim, hidden_dim, seed + 1000))
        self.register_buffer('W_up', create_fixed_random(atom_dim, hidden_dim, seed + 2000))

    def forward(self, combined_atom, x):
        x_proj = x @ self.W_proj.T
        gate = torch.sigmoid(x @ self.W_gate.T)
        up = x @ self.W_up.T

        modulation = (combined_atom.unsqueeze(-1) * torch.randn(self.atom_dim, self.hidden_dim, device=x.device).unsqueeze(0)).sum(dim=1)
        out = gate * (x_proj * modulation.unsqueeze(-2)) + (1 - gate) * up
        return out


class SimpleGatedExpander(nn.Module):
    def __init__(self, atom_dim, hidden_dim, seed=42):
        super().__init__()
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim

        rng = torch.Generator()
        rng.manual_seed(seed)
        self.register_buffer('R_down', torch.randn(hidden_dim, atom_dim, generator=rng) / math.sqrt(atom_dim))
        self.register_buffer('R_up', torch.randn(atom_dim, hidden_dim, generator=rng) / math.sqrt(hidden_dim))

    def forward(self, combined_atom, x):
        x_proj = x @ self.R_down

        gated = x_proj * combined_atom.unsqueeze(-2)

        out = gated @ self.R_up

        return out


class DirectWeightExpander(nn.Module):
    def __init__(self, atom_dim, hidden_dim, rank=32, seed=42):
        super().__init__()
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.rank = rank

        rng = torch.Generator()
        rng.manual_seed(seed)
        left = torch.randn(hidden_dim, atom_dim, rank, generator=rng) / math.sqrt(atom_dim * rank)
        right = torch.randn(atom_dim, rank, hidden_dim, generator=rng) / math.sqrt(atom_dim * rank)

        self.register_buffer('left_basis', left)
        self.register_buffer('right_basis', right)

    def forward(self, combined_atom, x):
        B, T, D = x.shape

        atom_mod = combined_atom.unsqueeze(-1) * self.left_basis
        atom_mod = atom_mod.sum(dim=1)

        x_proj = x @ self.right_basis.unsqueeze(0)
        x_proj = x_proj * atom_mod.unsqueeze(1).unsqueeze(-1)
        out = x_proj.sum(dim=-1)

        return out
