import pytest
import torch
from flm_train.schedules import OptimizerSchedule, SpeedrunStageSchedule
from flm_train.types import (
  OptimizerScheduleConfig,
  SpeedrunScheduleConfig,
  SpeedrunStageConfig,
)


def test_optimizer_schedule_applies_warmup_cooldown_and_momentum() -> None:
  parameter = torch.nn.Parameter(torch.ones(()))
  optimizer = torch.optim.SGD(
    [parameter],
    lr=2.0,
    momentum=0.8,
    weight_decay=0.5,
  )
  schedule = OptimizerSchedule(
    optimizer,
    total_steps=10,
    config=OptimizerScheduleConfig(
      warmup_steps=2,
      cooldown_steps=3,
      final_lr_scale=0.1,
      momentum_start=0.8,
      momentum_end=0.95,
      momentum_warmup_steps=2,
      scale_weight_decay_with_lr=True,
    ),
  )

  first = schedule.apply(1)
  assert first.learning_rate_scale == 0.5
  assert first.momentum == pytest.approx(0.875)
  assert optimizer.param_groups[0]["lr"] == 1.0
  assert optimizer.param_groups[0]["weight_decay"] == 0.25
  assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.875)

  last = schedule.apply(10)
  assert last.learning_rate_scale == pytest.approx(0.1)
  assert last.momentum == 0.95
  assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
  assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)


def test_optimizer_schedule_preserves_group_specific_base_values() -> None:
  first = torch.nn.Parameter(torch.ones(()))
  second = torch.nn.Parameter(torch.ones(()))
  optimizer = torch.optim.SGD(
    [
      {"params": [first], "lr": 1.0, "weight_decay": 0.2},
      {"params": [second], "lr": 0.5, "weight_decay": 0.0},
    ]
  )
  schedule = OptimizerSchedule(
    optimizer,
    total_steps=2,
    config=OptimizerScheduleConfig(warmup_steps=2),
  )

  schedule.apply(1)

  assert [group["lr"] for group in optimizer.param_groups] == [0.5, 0.25]
  assert [group["weight_decay"] for group in optimizer.param_groups] == [0.2, 0.0]


def test_optimizer_schedule_validates_configuration() -> None:
  parameter = torch.nn.Parameter(torch.ones(()))
  optimizer = torch.optim.SGD([parameter], lr=1.0)

  with pytest.raises(ValueError, match="cannot exceed"):
    OptimizerSchedule(
      optimizer,
      total_steps=2,
      config=OptimizerScheduleConfig(warmup_steps=2, cooldown_steps=1),
    )

  with pytest.raises(ValueError, match="both be set"):
    OptimizerSchedule(
      optimizer,
      total_steps=2,
      config=OptimizerScheduleConfig(momentum_start=0.8),
    )


def test_speedrun_stage_schedule_resolves_transitions_and_untie() -> None:
  schedule = SpeedrunStageSchedule(
    total_steps=6,
    config=SpeedrunScheduleConfig(
      stages=(
        SpeedrunStageConfig(
          end_step=2,
          batch_size=2,
          seq_len=8,
          mtp_weights=(1.0, 0.5),
          short_window=2,
          long_window=4,
        ),
        SpeedrunStageConfig(
          end_step=6,
          batch_size=4,
          seq_len=16,
          learning_rate_scale=1.5,
          mtp_weights=(1.0,),
          short_window=4,
          long_window=8,
        ),
      ),
      untie_step=3,
    ),
  )

  first = schedule.state_at(1)
  transition = schedule.state_at(3)
  last = schedule.state_at(6)

  assert first is not None and first.index == 0 and first.starts_stage
  assert transition is not None and transition.index == 1
  assert transition.starts_stage and transition.should_untie
  assert transition.embeddings_untied
  assert last is not None and last.index == 1 and not last.starts_stage
  assert last.embeddings_untied


def test_speedrun_stage_schedule_validates_coverage() -> None:
  with pytest.raises(ValueError, match="cover all"):
    SpeedrunStageSchedule(
      total_steps=4,
      config=SpeedrunScheduleConfig(
        stages=(SpeedrunStageConfig(end_step=3),),
      ),
    )


def test_optimizer_schedule_applies_stage_multiplier() -> None:
  parameter = torch.nn.Parameter(torch.ones(()))
  optimizer = torch.optim.SGD([parameter], lr=2.0, weight_decay=0.5)
  schedule = OptimizerSchedule(
    optimizer,
    total_steps=2,
    config=OptimizerScheduleConfig(scale_weight_decay_with_lr=True),
  )

  state = schedule.apply(1, learning_rate_multiplier=1.5)

  assert state.learning_rate_scale == 1.5
  assert state.weight_decay_scale == 1.5
  assert optimizer.param_groups[0]["lr"] == 3.0
  assert optimizer.param_groups[0]["weight_decay"] == 0.75
