"""Composite optimizer helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


class CompositeOptimizer(torch.optim.Optimizer):
  """One optimizer facade backed by multiple child optimizers."""

  def __init__(self, optimizers: Sequence[torch.optim.Optimizer]) -> None:
    self.optimizers = tuple(optimizers)
    if not self.optimizers:
      raise ValueError("CompositeOptimizer requires at least one optimizer")

    param_groups = [
      group for optimizer in self.optimizers for group in optimizer.param_groups
    ]
    defaults: dict[str, object] = {}
    super().__init__(param_groups, defaults)
    self.param_groups = param_groups

  @property
  def state(self) -> dict[torch.Tensor, dict[str, object]]:  # type: ignore[override]
    merged: dict[torch.Tensor, dict[str, object]] = {}
    for optimizer in self.optimizers:
      merged.update(optimizer.state)
    return merged

  @state.setter
  def state(self, state: dict[torch.Tensor, dict[str, object]]) -> None:
    self._state = state

  @torch.no_grad()
  def step(self, closure: Callable[[], object] | None = None) -> object | None:
    loss = None
    for index, optimizer in enumerate(self.optimizers):
      if index == 0:
        loss = optimizer.step(closure)
      else:
        optimizer.step()
    return loss

  def zero_grad(self, set_to_none: bool = True) -> None:
    for optimizer in self.optimizers:
      optimizer.zero_grad(set_to_none=set_to_none)

  def state_dict(self) -> dict[str, object]:
    return {
      "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
    }

  def load_state_dict(self, state_dict: dict[str, object]) -> None:
    optimizer_states = state_dict["optimizers"]
    if not isinstance(optimizer_states, list):
      raise ValueError("CompositeOptimizer state requires optimizer states")
    if len(optimizer_states) != len(self.optimizers):
      raise ValueError(
        "CompositeOptimizer state has "
        f"{len(optimizer_states)} optimizers, expected {len(self.optimizers)}"
      )
    for optimizer, optimizer_state in zip(
      self.optimizers, optimizer_states, strict=True
    ):
      optimizer.load_state_dict(optimizer_state)
