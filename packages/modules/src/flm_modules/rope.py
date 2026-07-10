"""Rotary position embeddings."""

from __future__ import annotations

import math
from enum import StrEnum

import torch
from torch import nn


class RopeLayout(StrEnum):
  LLAMA = "llama"
  INTERLEAVED = "interleaved"
  DEEPSEEK_V32 = "deepseek_v32"


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
  layout = RopeLayout(layout)
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
  if layout == RopeLayout.DEEPSEEK_V32:
    cos = cos[..., : rotary_dim // 2]
    sin = sin[..., : rotary_dim // 2]
    x1, x2 = rope[..., 0::2], rope[..., 1::2]
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
  else:
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


class SpeedrunYaRN(nn.Module):
  """Window-transition YaRN and half-truncated RoPE used by the speedrun."""

  def __init__(
    self,
    head_dim: int,
    max_seq_len: int,
    *,
    paired: bool = False,
  ) -> None:
    super().__init__()
    if head_dim % 4:
      raise ValueError("speedrun YaRN head_dim must be divisible by 4")
    self.head_dim = head_dim
    self.max_seq_len = max_seq_len
    self.paired = paired
    angular_freq = (1 / 1024) ** torch.linspace(
      0,
      1,
      steps=head_dim // 4,
      dtype=torch.float32,
    )
    angular_freq = angular_freq.repeat_interleave(2)
    angular_freq = torch.cat((angular_freq, angular_freq.new_zeros(head_dim // 2)))
    self.register_buffer("angular_freq", angular_freq)
    self.attention_scale = 0.1

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[-2]
    if seq_len > 2 * self.max_seq_len:
      raise ValueError("sequence exceeds speedrun YaRN capacity")
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    if self.paired:
      even = 2 * positions
      odd = even + 1
      theta = torch.cat(
        (
          torch.outer(even, self.angular_freq),
          torch.outer(odd, self.angular_freq),
        ),
        dim=-1,
      )
    else:
      theta = torch.outer(positions, self.angular_freq)
    factor1 = theta.cos().to(x.dtype).view(1, 1, seq_len, -1)
    factor2 = theta.sin().to(x.dtype)
    factor2[..., 1::2] *= -1
    factor2 = factor2.view(1, 1, seq_len, -1)
    flipped = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1)
    flipped = flipped.reshape_as(x)
    return factor1 * x + factor2 * flipped

  @torch.no_grad()
  def apply_window_change(
    self,
    old_window: int,
    new_window: int,
    *,
    alpha: int = 1,
    beta: int = 32,
  ) -> None:
    if old_window < 1 or new_window < 1:
      raise ValueError("YaRN windows must be positive")
    rotations = old_window * self.angular_freq / (2 * torch.pi)
    scaling_factor = old_window / new_window
    interpolation_weight = ((rotations - alpha) / (beta - alpha)).clamp(0, 1)
    self.angular_freq.mul_(scaling_factor + interpolation_weight * (1 - scaling_factor))
    self.attention_scale *= 0.2 * math.log(new_window / old_window) + 1
