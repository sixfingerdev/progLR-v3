"""ProgLR causal LM with V3-Embed, for HuggingFace transformers (trust_remote_code=True).

Architecture mirrors the training code exactly:
  hidden = V3Embed(x) + pos[:, :T]
  hidden = StdLayer (dense attention+FFN) then LRLayer (low-rank attention+FFN)
  head   = x @ R.T + (a/r) * (x @ V) @ U.T
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.utils import add_start_docstrings


class ProgLRConfig(PretrainedConfig):
    model_type = 'proglr'

    def __init__(self, d=768, heads=12, n_std=1, n_lr=1, r=16, a=64,
                 max_pos=1024, vocab_size=250000,
                 pad_token_id=0, unk_token_id=1, bos_token_id=2,
                 model_step=None, best_val_ppl=None, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)
        self.d = d
        self.heads = heads
        self.n_std = n_std
        self.n_lr = n_lr
        self.num_hidden_layers = n_std + n_lr
        self.r = r
        self.a = a
        self.max_pos = max_pos
        self.vocab_size = vocab_size
        self.unk_token_id = unk_token_id
        self.model_step = model_step
        self.best_val_ppl = best_val_ppl


class _StdLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_O = nn.Linear(dim, dim, bias=False)
        self.W1 = nn.Linear(dim, dim * 4, bias=False)
        self.W2 = nn.Linear(dim * 4, dim, bias=False)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape
        h, dh = self.h, self.dh
        n = self.n1(x)
        Q = self.W_Q(n).view(B, T, h, dh).transpose(1, 2)
        K = self.W_K(n).view(B, T, h, dh).transpose(1, 2)
        V = self.W_V(n).view(B, T, h, dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        x = x + self.W_O(a.transpose(1, 2).reshape(B, T, D))
        n = self.n2(x)
        return x + self.W2(F.relu(self.W1(n)))


class _LRLayer(nn.Module):
    """Low-rank attention + FFN, W = (a/r) U @ V^T (no full matrix materialized)."""

    def __init__(self, dim, heads, r=16, a=64):
        super().__init__()
        self.h, self.dh, self.r, self.a = heads, dim // heads, r, a
        for nm, od, id_ in [('q', dim, dim), ('k', dim, dim), ('v', dim, dim),
                            ('o', dim, dim), ('ff1', 4 * dim, dim), ('ff2', dim, 4 * dim)]:
            setattr(self, f'U_{nm}', nn.Parameter(torch.randn(od, r) * 0.02))
            setattr(self, f'V_{nm}', nn.Parameter(torch.randn(id_, r) * 0.02))
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def _l(self, x, nm):
        s = self.a / self.r
        return s * ((x @ getattr(self, f'V_{nm}')) @ getattr(self, f'U_{nm}').t())

    def forward(self, x):
        B, T, D = x.shape
        h, dh = self.h, self.dh
        n = self.n1(x)
        Q = self._l(n, 'q').view(B, T, h, dh).transpose(1, 2)
        K = self._l(n, 'k').view(B, T, h, dh).transpose(1, 2)
        V = self._l(n, 'v').view(B, T, h, dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        x = x + self._l(a.transpose(1, 2).reshape(B, T, D), 'o')
        n2 = self.n2(x)
        return x + self._l(F.relu(self._l(n2, 'ff1')), 'ff2')


class _V3Embed(nn.Module):
    """W = R + (a/r) U @ V^T. R is a frozen random buffer (500x embedding compression)."""

    def __init__(self, vocab_size, dim, rank=16, alpha=64):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        R = torch.randn(vocab_size, dim) / math.sqrt(dim)
        self.register_buffer('R', R)
        self.U = nn.Parameter(torch.randn(vocab_size, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(dim, rank) * 0.01)

    def forward_embed(self, input_ids):
        R_emb = F.embedding(input_ids, self.R)
        lora = (self.alpha / self.rank) * (F.embedding(input_ids, self.U) @ self.V.t())
        return R_emb + lora

    def forward_head(self, x):
        s = self.alpha / self.rank
        return F.linear(x, self.R) + s * ((x @ self.V) @ self.U.t())


@add_start_docstrings("ProgLR: V3-Embed + stacked Std/LR layers (hidden-state backbone).")
class ProgLRModel(PreTrainedModel):
    config_class = ProgLRConfig
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _supports_gradient_checkpointing = True
    _no_split_modules = ['_StdLayer', '_LRLayer']

    def __init__(self, config: ProgLRConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        self.emb = _V3Embed(config.vocab_size, config.d, config.r, config.a)
        self.pos = nn.Parameter(torch.randn(1, config.max_pos, config.d) * 0.01)
        self.layers = nn.ModuleList()
        for _ in range(config.n_std):
            self.layers.append(_StdLayer(config.d, config.heads))
        for _ in range(config.n_lr):
            self.layers.append(_LRLayer(config.d, config.heads, config.r, config.a))
        self.norm = nn.LayerNorm(config.d)
        self.post_init()

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, **kwargs):
        B, T = input_ids.shape
        x = self.emb.forward_embed(input_ids) + self.pos[:, :T]
        for l in self.layers:
            x = l(x)
        x = self.norm(x)
        return ModelOutput(last_hidden_state=x, hidden_states=None)


class ProgLRForCausalLM(ProgLRModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config: ProgLRConfig):
        super().__init__(config)

    def get_output_embeddings(self):
        return None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        out = ProgLRModel.forward(self, input_ids, attention_mask=attention_mask)
        logits = self.emb.forward_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1), reduction='sum')
        return ModelOutput(loss=loss, logits=logits, last_hidden_state=out.last_hidden_state)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None,
                                      past_key_values=None, **kwargs):
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
