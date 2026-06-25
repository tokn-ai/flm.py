"""Feed-forward layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
  def __init__(
    self,
    d_model: int,
    hidden_dim: int,
    dropout: float = 0.0,
    bias: bool = False,
  ) -> None:
    super().__init__()
    self.up = nn.Linear(d_model, 2 * hidden_dim, bias=bias)
    self.down = nn.Linear(hidden_dim, d_model, bias=bias)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate, value = self.up(x).chunk(2, dim=-1)
    x = F.silu(gate) * value
    return self.down(self.dropout(x))
