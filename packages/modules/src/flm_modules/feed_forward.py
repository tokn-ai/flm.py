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


class ReLUSquared(nn.Module):
  """Two-layer MLP using the ReLU-squared activation.

  The output projection can be zero-initialized so a newly constructed
  residual block starts as the identity. This is the eager reference form of
  the fused ReLU-squared MLP used by the nanoGPT speedrun.
  """

  def __init__(
    self,
    d_model: int,
    d_ff: int,
    bias: bool = False,
    *,
    zero_init_down: bool = False,
  ) -> None:
    super().__init__()
    self.up = nn.Linear(d_model, d_ff, bias=bias)
    self.down = nn.Linear(d_ff, d_model, bias=bias)
    if zero_init_down:
      nn.init.zeros_(self.down.weight)
      if self.down.bias is not None:
        nn.init.zeros_(self.down.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down(F.relu(self.up(x)).square())
