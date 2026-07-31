import torch
import torch.nn.functional as F

class AtomPool:
    def __init__(
        self,
        n_atoms: int,
        atom_dim: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        self.n_atoms = n_atoms
        self.atom_dim = atom_dim
        self.device = device
        self.dtype = dtype

        rng = torch.Generator()
        rng.manual_seed(seed)
        pool = torch.randn(n_atoms, atom_dim, generator=rng, dtype=dtype)
        pool = pool / pool.norm(dim=1, keepdim=True).clamp(min=1e-8)

        self.register_buffer("pool", pool)

    def register_buffer(self, name: str, tensor: torch.Tensor):
        self.__dict__[name] = tensor.to(self.device)

    def query(self, code: torch.Tensor) -> torch.Tensor:
        pool = self.pool.to(code.device)
        scores = code @ pool.T
        return scores

    def sample(self, scores: torch.Tensor, k: int, tau: float = 0.5, hard: bool = True):
        if hard:
            weights = F.gumbel_softmax(scores, tau=tau, hard=True, dim=-1)
        else:
            weights = F.gumbel_softmax(scores, tau=tau, hard=False, dim=-1)
        return weights

    def synthesize_weight(
        self,
        weights: torch.Tensor,
        expansion_fn,
        *expansion_args,
    ) -> torch.Tensor:
        pool = self.pool.to(weights.device)
        weighted_atoms = weights @ pool
        return expansion_fn(weighted_atoms, *expansion_args)

    def to(self, device: torch.device):
        self.pool = self.pool.to(device)
        self.device = device
        return self
