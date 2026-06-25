"""Rotary position embeddings."""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1, x2 = x[..., ::2], x[..., 1::2]
  return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(
  x: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
) -> torch.Tensor:
  cos = cos.unsqueeze(0).unsqueeze(0)
  sin = sin.unsqueeze(0).unsqueeze(0)
  return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
  def __init__(self, dim: int, base: float = 10_000.0) -> None:
    super().__init__()
    if dim % 2 != 0:
      raise ValueError("RoPE head dimension must be even")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.shape[-2]
    if positions is None:
      positions = torch.arange(seq_len, device=q.device)
    freqs = torch.outer(positions.to(self.inv_freq.dtype), self.inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    cos = emb.cos().to(dtype=q.dtype)
    sin = emb.sin().to(dtype=q.dtype)
    return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
