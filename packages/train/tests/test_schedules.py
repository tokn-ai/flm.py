import pytest
import torch
from flm_train.schedules import OptimizerSchedule
from flm_train.types import OptimizerScheduleConfig


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
