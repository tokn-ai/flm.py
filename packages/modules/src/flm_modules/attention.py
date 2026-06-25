"""Attention layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from flm_modules.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_heads: int,
    dropout: float = 0.0,
    bias: bool = False,
    rope_base: float = 10_000.0,
  ) -> None:
    super().__init__()
    if d_model % n_heads != 0:
      raise ValueError("d_model must be divisible by n_heads")

    self.d_model = d_model
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.dropout = dropout

    self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
    self.out = nn.Linear(d_model, d_model, bias=bias)
    self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    q, k = self.rope(q, k)
    y = F.scaled_dot_product_attention(
      q,
      k,
      v,
      attn_mask=None,
      dropout_p=self.dropout if self.training else 0.0,
      is_causal=True,
    )
    y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    return self.out(y)
