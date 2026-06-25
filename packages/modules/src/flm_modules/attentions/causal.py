"""Causal self-attention layers."""

from __future__ import annotations

from enum import StrEnum

import torch
from torch import nn
from torch.nn import functional as F

from flm_modules.rope import RotaryEmbedding


class AttentionBackend(StrEnum):
  TORCH = "torch"
  FLASH_ATTENTION2 = "flash_attention2"
  TILELANG = "tilelang"


class CausalSelfAttention(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_heads: int,
    bias: bool = False,
    rope_base: float = 10_000.0,
    backend: AttentionBackend | str = AttentionBackend.TORCH,
  ) -> None:
    super().__init__()
    if d_model % n_heads != 0:
      raise ValueError("d_model must be divisible by n_heads")

    self.d_model = d_model
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.backend = AttentionBackend(backend)

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
    if self.backend == AttentionBackend.FLASH_ATTENTION2:
      y = self._flash_attention2(q, k, v)
      return self.out(y)
    if self.backend == AttentionBackend.TILELANG:
      y = self._tilelang_attention(q, k, v)
      return self.out(y)

    y = F.scaled_dot_product_attention(
      q,
      k,
      v,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=True,
    )
    y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    return self.out(y)

  def _flash_attention2(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
  ) -> torch.Tensor:
    try:
      from flash_attn import flash_attn_func
    except ImportError as exc:
      raise ImportError(
        "flash_attention2 backend requires the flash-attn package"
      ) from exc

    batch_size, _, seq_len, _ = q.shape
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    y = flash_attn_func(q, k, v, causal=True)
    return y.contiguous().view(batch_size, seq_len, self.d_model)

  def _tilelang_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
  ) -> torch.Tensor:
    from flm_modules.kernels.tilelang.flash_attention import tilelang_flash_attention

    y = tilelang_flash_attention(q, k, v)
    batch_size, _, seq_len, _ = q.shape
    return y.contiguous().view(batch_size, seq_len, self.d_model)
