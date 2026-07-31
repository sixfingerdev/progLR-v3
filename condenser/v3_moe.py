"""
V3-MoE: Mixture of Random Experts for each layer.
Each expert has a different random R matrix.
Router learns to pick the right R per token.
Output = sum_k alpha_k(x) * (x @ R_k^T) + x @ LoRA^T
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class V3MoELayer(nn.Module):
    """Single V3-MoE layer: K random projections + shared LoRA + router."""
    def __init__(self, hidden_dim, num_heads=8, rank=8, num_experts=4, alpha=1.0, seed=0, layer_id=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.rank = rank
        self.num_experts = num_experts
        self.alpha = alpha

        gen = torch.Generator().manual_seed(42 + layer_id)

        # Generate K random R matrices for each of 6 weight matrices (Q, K, V, O, FF1, FF2)
        # Using register_buffer for fixed random matrices
        Rs = {}
        for name, shape in [('q', (hidden_dim, hidden_dim)), ('k', (hidden_dim, hidden_dim)),
                           ('v', (hidden_dim, hidden_dim)), ('o', (hidden_dim, hidden_dim)),
                           ('ff1', (4*hidden_dim, hidden_dim)), ('ff2', (hidden_dim, 4*hidden_dim))]:
            mats = torch.randn(num_experts, *shape, generator=gen) / math.sqrt(shape[-1])
            Rs[name] = mats
        self.Rs = nn.ParameterDict({k: nn.Parameter(v, requires_grad=False) for k, v in Rs.items()})

        # Shared LoRA (one per weight matrix, shared across experts)
        def make_lora(d_out, d_in):
            U = nn.Parameter(torch.randn(d_out, rank) * 0.01)
            V = nn.Parameter(torch.randn(d_in, rank) * 0.01)
            return U, V
        self.lora_q = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, rank)*0.01), nn.Parameter(torch.randn(hidden_dim, rank)*0.01)])
        self.lora_k = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, rank)*0.01), nn.Parameter(torch.randn(hidden_dim, rank)*0.01)])
        self.lora_v = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, rank)*0.01), nn.Parameter(torch.randn(hidden_dim, rank)*0.01)])
        self.lora_o = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, rank)*0.01), nn.Parameter(torch.randn(hidden_dim, rank)*0.01)])
        self.lora_ff1 = nn.ParameterList([nn.Parameter(torch.randn(4*hidden_dim, rank)*0.01), nn.Parameter(torch.randn(hidden_dim, rank)*0.01)])
        self.lora_ff2 = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, rank)*0.01), nn.Parameter(torch.randn(4*hidden_dim, rank)*0.01)])

        # Router: D -> D/4 -> K
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_experts),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def _expert_linear(self, x, R_stack, lora, route_w=None):
        """Compute output with mixture of experts."""
        B, T = x.shape[0], x.shape[1]
        K = self.num_experts

        x_R = torch.einsum('btd,kod->btko', x, R_stack)

        if route_w is None:
            route_w = F.softmax(self.router(x), dim=-1)
        expert_out = torch.einsum('btk,btko->bto', route_w, x_R)

        U, V = lora
        scaling = self.alpha / self.rank
        lora_out = scaling * (x @ V) @ U.T

        return expert_out + lora_out

    def forward(self, x):
        B, T, D = x.shape

        residual = x
        h = self.norm1(x)
        route_w = F.softmax(self.router(h), dim=-1)

        Q = self._expert_linear(h, self.Rs['q'], self.lora_q, route_w)
        K = self._expert_linear(h, self.Rs['k'], self.lora_k, route_w)
        V = self._expert_linear(h, self.Rs['v'], self.lora_v, route_w)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn_out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        attn_out = self._expert_linear(attn_out, self.Rs['o'], self.lora_o, route_w)
        x = residual + attn_out

        residual = x
        h = self.norm2(x)
        route_w2 = F.softmax(self.router(h), dim=-1)
        ff1 = self._expert_linear(h, self.Rs['ff1'], self.lora_ff1, route_w2)
        ff1 = F.relu(ff1)
        ff2 = self._expert_linear(ff1, self.Rs['ff2'], self.lora_ff2, route_w2)
        x = residual + ff2
        return x

    def count_trained(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class V3MoETransformer(nn.Module):
    """Full transformer with V3-MoE layers."""
    def __init__(self, vocab_size=50000, hidden_dim=768, num_layers=12,
                 num_heads=12, rank=8, alpha=1.0, num_experts=4):
        super().__init__()
        from condenser.v3_embed import V3WeightEmbedding
        self.v3_embed = V3WeightEmbedding(vocab_size, hidden_dim, rank, alpha)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.01)
        self.layers = nn.ModuleList([
            V3MoELayer(hidden_dim, num_heads, rank, num_experts, alpha, 42, i)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.v3_embed.forward_embed(input_ids) + self.pos_embedding[:, :T]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.v3_embed.forward_head(x)

    def count_trained(self):
        return (sum(p.numel() for p in self.v3_embed.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.pos_embedding if p.requires_grad) +
                sum(l.count_trained() for l in self.layers) +
                sum(p.numel() for p in self.norm.parameters() if p.requires_grad))

    def count_total(self):
        return sum(p.numel() for p in self.parameters())
