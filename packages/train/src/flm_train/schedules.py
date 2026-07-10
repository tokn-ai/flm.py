"""Step-dependent optimizer schedules for speedrun training recipes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flm_train.types import OptimizerScheduleConfig


@dataclass(frozen=True)
class OptimizerScheduleState:
  learning_rate_scale: float
  momentum: float | None
  weight_decay_scale: float


class OptimizerSchedule:
  """Apply warmup/cooldown policies without owning optimizer state."""

  def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    config: OptimizerScheduleConfig,
  ) -> None:
    if total_steps < 1:
      raise ValueError("total_steps must be positive")
    if config.warmup_steps < 0 or config.cooldown_steps < 0:
      raise ValueError("schedule step counts must be non-negative")
    if config.warmup_steps + config.cooldown_steps > total_steps:
      raise ValueError("warmup and cooldown cannot exceed total steps")
    if not 0 <= config.final_lr_scale <= 1:
      raise ValueError("final_lr_scale must be in [0, 1]")
    if config.momentum_warmup_steps < 0:
      raise ValueError("momentum_warmup_steps must be non-negative")
    for momentum in (config.momentum_start, config.momentum_end):
      if momentum is not None and not 0 <= momentum < 1:
        raise ValueError("scheduled momentum must be in [0, 1)")
    if (config.momentum_start is None) != (config.momentum_end is None):
      raise ValueError("momentum_start and momentum_end must both be set")

    self.optimizer = optimizer
    self.total_steps = total_steps
    self.config = config
    self._base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
    self._base_weight_decays = tuple(
      float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups
    )

  def apply(self, step: int) -> OptimizerScheduleState:
    state = self.state_at(step)
    for group, base_lr, base_weight_decay in zip(
      self.optimizer.param_groups,
      self._base_lrs,
      self._base_weight_decays,
      strict=True,
    ):
      group["lr"] = base_lr * state.learning_rate_scale
      group["weight_decay"] = base_weight_decay * state.weight_decay_scale
      if state.momentum is not None and "momentum" in group:
        group["momentum"] = state.momentum
    return state

  def state_at(self, step: int) -> OptimizerScheduleState:
    if not 1 <= step <= self.total_steps:
      raise ValueError(f"step must be in [1, {self.total_steps}]")
    learning_rate_scale = self._learning_rate_scale(step)
    weight_decay_scale = (
      learning_rate_scale if self.config.scale_weight_decay_with_lr else 1.0
    )
    return OptimizerScheduleState(
      learning_rate_scale=learning_rate_scale,
      momentum=self._momentum(step),
      weight_decay_scale=weight_decay_scale,
    )

  def _learning_rate_scale(self, step: int) -> float:
    if self.config.warmup_steps and step <= self.config.warmup_steps:
      return step / self.config.warmup_steps
    cooldown_start = self.total_steps - self.config.cooldown_steps
    if self.config.cooldown_steps and step > cooldown_start:
      progress = (step - cooldown_start) / self.config.cooldown_steps
      return 1.0 + progress * (self.config.final_lr_scale - 1.0)
    return 1.0

  def _momentum(self, step: int) -> float | None:
    start = self.config.momentum_start
    end = self.config.momentum_end
    if start is None and end is None:
      return None
    if start is None or end is None:  # guarded during construction
      return None
    warmup_steps = self.config.momentum_warmup_steps
    if warmup_steps == 0:
      return end
    progress = min(step / warmup_steps, 1.0)
    return start + progress * (end - start)
