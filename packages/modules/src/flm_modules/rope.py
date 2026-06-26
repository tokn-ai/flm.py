"""Rotary position embeddings."""

from __future__ import annotations

from enum import StrEnum

import torch
from torch import nn


class RopeLayout(StrEnum):
  LLAMA = "llama"
  INTERLEAVED = "interleaved"


def rotate_half(
  x: torch.Tensor,
  layout: RopeLayout | str = RopeLayout.LLAMA,
) -> torch.Tensor:
  layout = RopeLayout(layout)
  if layout == RopeLayout.INTERLEAVED:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
  x: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  layout: RopeLayout | str = RopeLayout.LLAMA,
  rotary_dim: int | None = None,
) -> torch.Tensor:
  rotary_dim = rotary_dim if rotary_dim is not None else cos.shape[-1]
  if rotary_dim <= 0:
    raise ValueError("rotary_dim must be positive")
  if rotary_dim > x.shape[-1]:
    raise ValueError("rotary_dim must not exceed the input hidden dimension")

  while cos.ndim < x.ndim:
    if cos.ndim == x.ndim - 1 and cos.shape[0] == x.shape[0]:
      cos = cos.unsqueeze(1)
      sin = sin.unsqueeze(1)
    else:
      cos = cos.unsqueeze(0)
      sin = sin.unsqueeze(0)

  nope, rope = x[..., :-rotary_dim], x[..., -rotary_dim:]
  rotated = (rope * cos) + (rotate_half(rope, layout=layout) * sin)
  if rotary_dim == x.shape[-1]:
    return rotated
  return torch.cat((nope, rotated), dim=-1)


class RotaryEmbedding(nn.Module):
  def __init__(
    self,
    dim: int,
    base: float = 10_000.0,
    layout: RopeLayout | str = RopeLayout.LLAMA,
  ) -> None:
    super().__init__()
    if dim % 2 != 0:
      raise ValueError("RoPE head dimension must be even")
    self.dim = dim
    self.layout = RopeLayout(layout)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def forward(
    self,
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = x.shape[-2]
    if positions is None:
      positions = torch.arange(seq_len, device=x.device)
    freqs = positions.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
    if self.layout == RopeLayout.INTERLEAVED:
      emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    else:
      emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=x.dtype)
    sin = emb.sin().to(dtype=x.dtype)
    return cos, sin
