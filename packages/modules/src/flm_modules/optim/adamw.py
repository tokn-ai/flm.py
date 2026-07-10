"""AdamW optimizer helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def configure_adamw(
  model: nn.Module,
  learning_rate: float = 3e-4,
  weight_decay: float = 0.1,
  betas: tuple[float, float] = (0.9, 0.95),
  eps: float = 1e-8,
) -> torch.optim.AdamW:
  decay: list[nn.Parameter] = []
  no_decay: list[nn.Parameter] = []

  for param in model.parameters():
    if not param.requires_grad:
      continue
    if param.ndim >= 2:
      decay.append(param)
    else:
      no_decay.append(param)

  param_groups: Iterable[dict[str, object]] = [
    {"params": decay, "weight_decay": weight_decay},
    {"params": no_decay, "weight_decay": 0.0},
  ]
  return torch.optim.AdamW(
    param_groups,
    lr=learning_rate,
    betas=betas,
    eps=eps,
  )
