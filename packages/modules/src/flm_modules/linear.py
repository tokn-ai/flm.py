"""Linear projection layers."""

from __future__ import annotations

import torch
from torch import nn


class GroupedLinear(nn.Linear):
  """Block-diagonal grouped linear projection.

  Inputs use shape ``(..., n_groups, in_features_per_group)``. The same public
  weight layout as ``nn.Linear`` is kept for easy compatibility with
  Transformers' DeepSeek V4 grouped output projection.
  """

  def __init__(
    self,
    in_features_per_group: int,
    out_features: int,
    n_groups: int,
    bias: bool = False,
  ) -> None:
    if n_groups <= 0:
      raise ValueError("n_groups must be positive")
    if out_features % n_groups != 0:
      raise ValueError("out_features must be divisible by n_groups")
    super().__init__(in_features_per_group, out_features, bias=bias)
    self.n_groups = n_groups

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_shape = x.shape[:-2]
    hidden_dim = x.shape[-1]
    if x.shape[-2] != self.n_groups:
      raise ValueError("input group dimension must match n_groups")
    if hidden_dim != self.in_features:
      raise ValueError("input hidden dimension must match in_features_per_group")

    weight = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
    x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
    y = torch.bmm(x, weight).transpose(0, 1)
    return y.reshape(*input_shape, self.n_groups, -1)
