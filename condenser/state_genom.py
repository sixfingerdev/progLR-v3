"""
StateGenom: Parametric Weight Generation for LaWS v2

The core insight: instead of storing individual weights for each layer,
a small recurrent "Genom" evolves a hidden state across layers and
generates each layer's weight on-the-fly.

Training: only Genom (few MB) is trained.
Inference: weights are generated sequentially, used once, then discarded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StateGenom(nn.Module):
    """
    Recurrent weight generator. Evolves state across layers.
    Generates each layer's weight matrix from a tiny trainable core.
    """
    def __init__(
        self,
        state_dim: int = 512,
        hidden_dim: int = 1024,
        n_layers_trainable: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.state_init = nn.Parameter(torch.randn(state_dim) * 0.02)

        self.state_update = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

        self.lora_A = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(state_dim, hidden_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def generate_weight(self, state: torch.Tensor) -> torch.Tensor:
        lora = self.lora_A @ self.lora_B
        W = lora
        return W

    def generate_weight_full(self, state: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
        upscale = nn.functional.linear(state, self.lora_A)
        W = upscale.unsqueeze(-1) * torch.randn(self.state_dim, out_dim, in_dim, device=state.device)
        W = W.sum(dim=0)
        return W

    def forward(self, n_layers: int, device: torch.device) -> list[torch.Tensor]:
        state = self.state_init.unsqueeze(0).expand(1, -1).to(device)
        weights = []

        for l in range(n_layers):
            delta = self.state_update(state)
            state = state + delta
            state = F.layer_norm(state, (self.state_dim,))

            W = self.generate_weight(state)
            weights.append(W)

        return weights


class LaWSv2Layer(nn.Module):
    """
    A single layer that uses StateGenom to generate its weight on-the-fly.
    The weight is used and discarded - never stored as a trainable parameter.
    """
    def __init__(self, hidden_dim: int, genom: StateGenom, layer_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_id = layer_id
        self.genom = genom
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, genom_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)

        delta = self.genom.state_update(genom_state)
        new_state = genom_state + delta
        new_state = F.layer_norm(new_state, (self.genom.state_dim,))

        W = self.genom.generate_weight_full(
            new_state, self.hidden_dim, self.hidden_dim
        )

        out = x @ W.T
        out = F.relu(out)

        return out + residual, new_state


class LaWSv2Model(nn.Module):
    """
    Full model using LaWS v2 with StateGenom weight generation.
    Only Genom parameters are trained - all weights are ephemeral.
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        state_dim: int = 512,
        genom_hidden: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.genom = StateGenom(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            LaWSv2Layer(hidden_dim, self.genom, i)
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.token_embedding(input_ids)

        state = self.genom.state_init.unsqueeze(0).expand(B, -1).to(x.device)

        for layer in self.layers:
            x, state = layer(x, state)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for b in self.buffers())
        return trainable, buffers

    def effective_params(self):
        return self.num_layers * self.hidden_dim * self.hidden_dim * 2

    def memory_analysis(self):
        trainable, _ = self.count_params()
        mem_fp16 = trainable * 2 / (1024**3)

        optimizer_mem = trainable * 4 / (1024**3)

        act_mem = self.hidden_dim * 512 * 2 / (1024**3)

        total = mem_fp16 + optimizer_mem * 2 + act_mem

        return {
            'trainable_params': trainable,
            'model_gb_fp16': mem_fp16,
            'optimizer_gb': optimizer_mem,
            'activations_gb': act_mem,
            'total_gb': total,
            'fits_in_6gb': total < 6.0,
        }
