"""Normalization layers."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(d_model))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return self.weight * x * scale
