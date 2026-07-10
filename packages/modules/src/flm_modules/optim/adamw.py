"""AdamW optimizer helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer


class CautiousAdamW(Optimizer):
  """AdamW whose decoupled decay is gated to agree with the Adam update."""

  def __init__(
    self,
    params,
    *,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    weight_decay_lr_power: int = 1,
  ) -> None:
    beta1, beta2 = betas
    if lr < 0:
      raise ValueError("learning rate must be non-negative")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
      raise ValueError("Adam betas must be in [0, 1)")
    if eps < 0 or weight_decay < 0:
      raise ValueError("eps and weight_decay must be non-negative")
    if weight_decay_lr_power not in {1, 2}:
      raise ValueError("weight_decay_lr_power must be 1 or 2")
    super().__init__(
      params,
      {
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "weight_decay_lr_power": weight_decay_lr_power,
      },
    )

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      beta1, beta2 = group["betas"]
      for param in group["params"]:
        if param.grad is None:
          continue
        if param.grad.is_sparse:
          raise RuntimeError("CautiousAdamW does not support sparse gradients")
        state = self.state[param]
        if not state:
          state["step"] = 0
          state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
          state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
        state["step"] += 1
        grad = param.grad.float()
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5)
        denominator.add_(group["eps"])
        update = exp_avg.div(denominator).div_(bias_correction1)
        if group["weight_decay"]:
          mask = (update * param.float()) >= 0
          param.add_(
            param * mask,
            alpha=-(
              group["lr"] ** group["weight_decay_lr_power"] * group["weight_decay"]
            ),
          )
        param.add_(update.to(param.dtype), alpha=-group["lr"])
    return loss


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
