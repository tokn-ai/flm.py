"""Feed-forward layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
  def __init__(
    self,
    d_model: int,
    d_ff: int,
    bias: bool = False,
  ) -> None:
    super().__init__()
    self.up = nn.Linear(d_model, 2 * d_ff, bias=bias)
    self.down = nn.Linear(d_ff, d_model, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate, value = self.up(x).chunk(2, dim=-1)
    x = F.silu(gate) * value
    return self.down(x)
