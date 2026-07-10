"""Muon optimizer helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from flm_modules.optim.composite import CompositeOptimizer


class Muon(Optimizer):
  """Muon optimizer for 2D matrix parameters."""

  def __init__(
    self,
    params: Iterable[dict[str, object]],
    *,
    lr: float = 3e-4,
    momentum: float = 0.95,
    weight_decay: float = 0.1,
    nesterov: bool = True,
    ns_steps: int = 5,
  ) -> None:
    if lr < 0.0:
      raise ValueError(f"invalid learning rate: {lr}")
    if momentum < 0.0:
      raise ValueError(f"invalid momentum value: {momentum}")
    if weight_decay < 0.0:
      raise ValueError(f"invalid weight_decay value: {weight_decay}")
    if ns_steps < 1:
      raise ValueError(f"invalid ns_steps value: {ns_steps}")

    defaults = {
      "lr": lr,
      "momentum": momentum,
      "weight_decay": weight_decay,
      "nesterov": nesterov,
      "ns_steps": ns_steps,
    }
    super().__init__(params, defaults)
    for group in self.param_groups:
      for param in group["params"]:
        if param.ndim != 2:
          raise ValueError(
            "Muon only supports 2D parameters, "
            f"but found parameter with shape {tuple(param.shape)}"
          )

  @torch.no_grad()
  def step(self, closure: Callable[[], object] | None = None) -> object | None:
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      self._muon_step(group)
    return loss

  def _muon_step(self, group: dict[str, object]) -> None:
    lr = float(group["lr"])
    momentum = float(group["momentum"])
    weight_decay = float(group["weight_decay"])
    nesterov = bool(group["nesterov"])
    ns_steps = int(group["ns_steps"])

    for param in group["params"]:
      if param.grad is None:
        continue
      grad = param.grad
      if grad.is_sparse:
        raise RuntimeError("Muon does not support sparse gradients")
      if weight_decay:
        param.mul_(1.0 - lr * weight_decay)

      state = self.state[param]
      if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(param)
      buffer = state["momentum_buffer"]
      buffer.lerp_(grad, 1.0 - momentum)
      update = grad.lerp(buffer, momentum) if nesterov else buffer
      update = _orthogonalize_update(update, ns_steps=ns_steps)
      param.add_(update, alpha=-_adjust_muon_lr(lr, param.shape))


def configure_muon(
  model: nn.Module,
  learning_rate: float = 3e-4,
  weight_decay: float = 0.1,
  momentum: float = 0.95,
  nesterov: bool = True,
  ns_steps: int = 5,
  adamw_betas: tuple[float, float] = (0.9, 0.95),
  adamw_eps: float = 1e-8,
) -> CompositeOptimizer:
  muon_params: list[nn.Parameter] = []
  adamw_params: list[nn.Parameter] = []

  for param in model.parameters():
    if not param.requires_grad:
      continue
    if param.ndim >= 2:
      muon_params.append(param)
    else:
      adamw_params.append(param)

  optimizers: list[torch.optim.Optimizer] = []
  if muon_params:
    optimizers.append(
      Muon(
        muon_params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        ns_steps=ns_steps,
      )
    )
  if adamw_params:
    optimizers.append(
      torch.optim.AdamW(
        [{"params": adamw_params, "weight_decay": 0.0}],
        lr=learning_rate,
        betas=adamw_betas,
        eps=adamw_eps,
      )
    )
  return CompositeOptimizer(optimizers)


def _orthogonalize_update(update: torch.Tensor, *, ns_steps: int) -> torch.Tensor:
  original_shape = update.shape
  matrix = update.flatten(1) if update.ndim > 2 else update
  orthogonalized = _zeroth_power_via_newton_schulz5(matrix, steps=ns_steps)
  return orthogonalized.reshape(original_shape)


def _zeroth_power_via_newton_schulz5(
  matrix: torch.Tensor, *, steps: int
) -> torch.Tensor:
  if matrix.ndim != 2:
    raise ValueError("Newton-Schulz orthogonalization expects a matrix")

  a, b, c = 3.4445, -4.7750, 2.0315
  original_dtype = matrix.dtype
  x = matrix.bfloat16()
  if x.size(0) > x.size(1):
    x = x.T

  x.div_(x.norm().clamp(min=1e-7))
  for _ in range(steps):
    xx_t = x @ x.T
    gram_update = torch.addmm(xx_t, xx_t, xx_t, beta=b, alpha=c)
    x = torch.addmm(x, gram_update, x, beta=a)

  if matrix.size(0) > matrix.size(1):
    x = x.T
  return x.to(dtype=original_dtype)


def _adjust_muon_lr(lr: float, shape: torch.Size) -> float:
  rows = shape[0]
  cols = math.prod(shape[1:])
  return lr * max(1.0, rows / cols) ** 0.5
