"""Step-dependent optimizer schedules for speedrun training recipes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flm_train.types import (
  OptimizerScheduleConfig,
  SpeedrunScheduleConfig,
  SpeedrunStageConfig,
)


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
    cooldown_end = config.cooldown_end_step or total_steps
    if not 1 <= cooldown_end <= total_steps:
      raise ValueError("cooldown_end_step must be within training steps")
    if config.warmup_steps + config.cooldown_steps > cooldown_end:
      raise ValueError("warmup and cooldown cannot exceed total steps")
    if not 0 <= config.final_lr_scale <= 1:
      raise ValueError("final_lr_scale must be in [0, 1]")
    if config.momentum_warmup_steps < 0 or config.momentum_cooldown_steps < 0:
      raise ValueError("momentum schedule steps must be non-negative")
    if config.momentum_warmup_steps + config.momentum_cooldown_steps > total_steps:
      raise ValueError("momentum warmup and cooldown cannot exceed total steps")
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

  def apply(
    self,
    step: int,
    *,
    learning_rate_multiplier: float = 1.0,
  ) -> OptimizerScheduleState:
    if learning_rate_multiplier <= 0:
      raise ValueError("learning_rate_multiplier must be positive")
    state = self.state_at(step)
    cooldown_progress = self._cooldown_progress(step)
    learning_rate_scale = (
      state.learning_rate_scale * learning_rate_multiplier
      if cooldown_progress == 0
      else learning_rate_multiplier * (1 - cooldown_progress)
      + self.config.final_lr_scale * cooldown_progress
    )
    weight_decay_scale = (
      learning_rate_scale if self.config.scale_weight_decay_with_lr else 1.0
    )
    state = OptimizerScheduleState(
      learning_rate_scale=learning_rate_scale,
      momentum=state.momentum,
      weight_decay_scale=weight_decay_scale,
    )
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
    progress = self._cooldown_progress(step)
    if progress:
      return 1.0 + progress * (self.config.final_lr_scale - 1.0)
    return 1.0

  def _cooldown_progress(self, step: int) -> float:
    if not self.config.cooldown_steps:
      return 0.0
    cooldown_end = self.config.cooldown_end_step or self.total_steps
    cooldown_start = cooldown_end - self.config.cooldown_steps
    if step <= cooldown_start:
      return 0.0
    return min((step - cooldown_start) / self.config.cooldown_steps, 1.0)

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
    if step <= warmup_steps:
      progress = (step - 1) / warmup_steps
      return start + progress * (end - start)
    cooldown_steps = self.config.momentum_cooldown_steps
    cooldown_start = self.total_steps - cooldown_steps
    if cooldown_steps and step > cooldown_start:
      progress = (step - cooldown_start) / cooldown_steps
      return end + progress * (start - end)
    return end


@dataclass(frozen=True)
class SpeedrunStageState:
  index: int
  stage: SpeedrunStageConfig
  starts_stage: bool
  should_untie: bool
  embeddings_untied: bool
  mtp_weights: tuple[float, ...] | None


class SpeedrunStageSchedule:
  """Resolve discrete algorithm and input-shape transitions by training step."""

  def __init__(
    self,
    *,
    total_steps: int,
    config: SpeedrunScheduleConfig,
  ) -> None:
    if total_steps < 1:
      raise ValueError("total_steps must be positive")
    self.total_steps = total_steps
    self.config = config
    previous_end = 0
    for stage in config.stages:
      if stage.end_step <= previous_end:
        raise ValueError("speedrun stage end steps must be strictly increasing")
      if stage.batch_size is not None and stage.batch_size < 1:
        raise ValueError("stage batch_size must be positive")
      if stage.seq_len is not None and stage.seq_len < 1:
        raise ValueError("stage seq_len must be positive")
      if stage.learning_rate_scale <= 0:
        raise ValueError("stage learning_rate_scale must be positive")
      if stage.mtp_weights is not None and (
        not stage.mtp_weights
        or stage.mtp_weights[0] <= 0
        or any(weight < 0 for weight in stage.mtp_weights)
      ):
        raise ValueError("stage mtp_weights are invalid")
      if stage.mtp_weights_end is not None:
        if stage.mtp_weights is None:
          raise ValueError("stage mtp_weights_end requires mtp_weights")
        if len(stage.mtp_weights_end) != len(stage.mtp_weights):
          raise ValueError("stage MTP weight endpoints must have equal lengths")
        if (
          not stage.mtp_weights_end
          or stage.mtp_weights_end[0] <= 0
          or any(weight < 0 for weight in stage.mtp_weights_end)
        ):
          raise ValueError("stage mtp_weights_end is invalid")
      if (stage.short_window is None) != (stage.long_window is None):
        raise ValueError("stage attention windows must both be set")
      for window in (stage.short_window, stage.long_window):
        if window is not None and window < 1:
          raise ValueError("stage attention windows must be positive")
      previous_end = stage.end_step
    if config.stages and config.stages[-1].end_step < total_steps:
      raise ValueError("final speedrun stage must cover all training steps")
    if config.untie_step is not None and not config.stages:
      raise ValueError("untie_step requires at least one speedrun stage")
    if config.untie_step is not None and not 1 <= config.untie_step <= total_steps:
      raise ValueError("untie_step must be within training steps")
    if (config.final_eval_short_window is None) != (
      config.final_eval_long_window is None
    ):
      raise ValueError("final eval attention windows must both be set")
    if config.final_eval_short_window is not None and not (
      1 <= config.final_eval_short_window <= config.final_eval_long_window
    ):
      raise ValueError("final eval attention windows are invalid")

  def state_at(self, step: int) -> SpeedrunStageState | None:
    if not 1 <= step <= self.total_steps:
      raise ValueError(f"step must be in [1, {self.total_steps}]")
    for index, stage in enumerate(self.config.stages):
      previous_end = 0 if index == 0 else self.config.stages[index - 1].end_step
      if step <= stage.end_step:
        duration = stage.end_step - previous_end
        progress = (step - previous_end - 1) / duration
        mtp_weights = stage.mtp_weights
        if mtp_weights is not None and stage.mtp_weights_end is not None:
          mtp_weights = tuple(
            start + progress * (end - start)
            for start, end in zip(
              mtp_weights,
              stage.mtp_weights_end,
              strict=True,
            )
          )
        return SpeedrunStageState(
          index=index,
          stage=stage,
          starts_stage=step == previous_end + 1,
          should_untie=step == self.config.untie_step,
          embeddings_untied=(
            self.config.untie_step is not None and step >= self.config.untie_step
          ),
          mtp_weights=mtp_weights,
        )
    return None
