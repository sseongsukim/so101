"""Transformer / MLP building blocks.

Ported from simdist's `simdist/modeling/modules.py` (flax nnx) to pure PyTorch.
Kept faithful down to activation flavour, init distribution and LayerNorm eps so
the torch model trains like the JAX original:

* `nnx.gelu` is `jax.nn.gelu(approximate=True)` -> `nn.GELU(approximate="tanh")`.
* `nnx.Linear` defaults to `lecun_normal()` kernels and zero bias -> see
  `lecun_normal_`. torch's own default (kaiming_uniform) is *not* the same.
* `nnx.LayerNorm` defaults to eps 1e-6, torch to 1e-5.
* Residual blocks are post-LN: `LN(x + sublayer(x))`.

Not ported: `CNN` / `TransposeCNN` / `EmptyModule` — only used by the go2
height-scan encoder, which this package drops (see `encoder.py`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

_TRUNC_NORM_STD_CORRECTION = 0.8796256610342398  # flax's constant for [-2, 2]


def lecun_normal_(weight: torch.Tensor) -> None:
    """In-place flax `lecun_normal()`: truncated normal, fan_in scaling.

    flax draws from a truncated normal on [-2, 2] and divides the stddev by
    0.8796... so the *post-truncation* variance is still 1/fan_in.
    """
    fan_in = weight.shape[-1]
    std = math.sqrt(1.0 / fan_in) / _TRUNC_NORM_STD_CORRECTION
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def linear(in_features: int, out_features: int) -> nn.Linear:
    """`nn.Linear` with flax's default init (lecun_normal kernel, zero bias)."""
    layer = nn.Linear(in_features, out_features)
    lecun_normal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _multihead_attention(embed_dim: int, num_heads: int, dropout: float) -> nn.MultiheadAttention:
    """`nnx.MultiHeadAttention` equivalent: qkv/out projections of width
    `embed_dim`, lecun_normal kernels, zero biases, dropout on attention weights."""
    attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    # in_proj_weight is (3 * embed_dim, embed_dim): fan_in is embed_dim for all
    # three projections, matching flax's per-projection lecun_normal.
    lecun_normal_(attn.in_proj_weight)
    nn.init.zeros_(attn.in_proj_bias)
    lecun_normal_(attn.out_proj.weight)
    nn.init.zeros_(attn.out_proj.bias)
    return attn


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Bool mask for `nn.MultiheadAttention`: True == blocked, so position i may
    only attend to j <= i (same as `nnx.make_causal_mask`)."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)


class MLP(nn.Module):
    """`hidden_dims` GELU layers then a linear output layer.

    With `hidden_dims=[]` this is a single bare `nn.Linear` — that is what the
    original's `*_layers: 0` config entries mean.

    Faithful quirk: simdist builds `len(hidden_dims) - 1` dropout layers and
    applies `dropout[i]` only for `i < len(layers) - 1`. So an MLP with one
    hidden layer has *no* dropout at all, which makes the `dropout.mlp: 0.2` in
    the original config dead for every transformer FFN (they all use exactly one
    hidden layer). Reproduced rather than "fixed" so behaviour matches.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.dropouts = nn.ModuleList(
            nn.Dropout(dropout_rate) for _ in range(max(0, len(hidden_dims) - 1))
        )
        self.output_layer = linear(prev_dim, output_dim)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            if i < len(self.layers) - 1:
                x = self.dropouts[i](x)
        return self.output_layer(x)


class Embedding(nn.Module):
    """Projection to `embed_dim` plus a learned positional encoding.

    `pos_emb` is initialised from a *unit* normal, as in the original
    (`jax.random.normal(...)` with no scaling).
    """

    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        hidden_dims: Sequence[int],
        embed_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.enc = MLP(input_dim, hidden_dims, embed_dim, dropout_rate)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x) + self.pos_emb


class TransformerEncoderLayer(nn.Module):
    """Post-LN self-attention block."""

    def __init__(
        self,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.attention = _multihead_attention(embed_dim, num_heads, attention_dropout_rate)
        self.dense_proj = MLP(embed_dim, [mlp_hidden_dim], embed_dim, mlp_dropout_rate)
        self.layernorm_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = causal_mask(x.shape[1], x.device) if self.causal else None
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        proj_input = self.layernorm_1(x + attn_out)
        return self.layernorm_2(proj_input + self.dense_proj(proj_input))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                num_heads=num_heads,
                attention_dropout_rate=attention_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
                causal=causal,
            )
            for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Post-LN self-attention (optionally causal) + cross-attention block.

    Cross-attention is never masked in the world model (the original always
    passed `mask=None` for it), so no padding-mask path is ported.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.attention_1 = _multihead_attention(embed_dim, num_heads, attention_dropout_rate)
        self.attention_2 = _multihead_attention(embed_dim, num_heads, attention_dropout_rate)
        self.dense_proj = MLP(embed_dim, [mlp_hidden_dim], embed_dim, mlp_dropout_rate)
        self.layernorm_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_3 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        mask = causal_mask(x.shape[1], x.device) if self.causal else None
        self_attn, _ = self.attention_1(x, x, x, attn_mask=mask, need_weights=False)
        out_1 = self.layernorm_1(x + self_attn)
        cross_attn, _ = self.attention_2(
            out_1, encoder_outputs, encoder_outputs, need_weights=False
        )
        out_2 = self.layernorm_2(out_1 + cross_attn)
        return self.layernorm_3(out_2 + self.dense_proj(out_2))


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                num_heads=num_heads,
                attention_dropout_rate=attention_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
                causal=causal,
            )
            for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_outputs)
        return x
