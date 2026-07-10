"""Normalization layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(d_model))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    scale = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
    return (self.weight * x * scale).to(dtype)


class LayerNorm(nn.Module):
  """
  Layer Normalization.
  """

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
    self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

  def forward(self, x: torch.Tensor):
    return F.layer_norm(
      x.float(), (self.dim,), self.weight, self.bias, self.eps
    ).type_as(x)
