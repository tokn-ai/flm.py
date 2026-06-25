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
) -> torch.Tensor:
  cos = cos.unsqueeze(0).unsqueeze(0)
  sin = sin.unsqueeze(0).unsqueeze(0)
  return (x * cos) + (rotate_half(x, layout=layout) * sin)


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
    self.layout = RopeLayout(layout)
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
    if self.layout == RopeLayout.INTERLEAVED:
      emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    else:
      emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=q.dtype)
    sin = emb.sin().to(dtype=q.dtype)
    return (
      apply_rotary(q, cos, sin, layout=self.layout),
      apply_rotary(k, cos, sin, layout=self.layout),
    )
