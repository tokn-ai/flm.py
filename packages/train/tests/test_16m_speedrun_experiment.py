from dataclasses import replace
from pathlib import Path

from flm_train.config import load_experiment_config
from flm_train.models import build_model
from flm_train.presets import _config_with_resolved_intervals

FULL_CONFIG = Path("experiments/16m_fineweb_speedrun.yaml")
SMOKE_CONFIG = Path("experiments/16m_fineweb_speedrun_smoke.yaml")


def test_16m_speedrun_experiment_has_expected_parameter_count() -> None:
  config = load_experiment_config(FULL_CONFIG)

  model = build_model(config.to_train_config(), vocab_size=8192)

  assert sum(parameter.numel() for parameter in model.parameters()) == 15_931_066


def test_16m_speedrun_smoke_uses_the_same_model() -> None:
  full = load_experiment_config(FULL_CONFIG)
  smoke = load_experiment_config(SMOKE_CONFIG)

  assert smoke.model == full.model
  assert smoke.data.dataset_root == full.data.dataset_root
  assert smoke.data.encoding_name == full.data.encoding_name


def test_16m_speedrun_schedule_matches_reference_token_budget() -> None:
  config = load_experiment_config(FULL_CONFIG)
  previous_end = 0
  token_budget = 0
  for stage in config.speedrun_schedule.stages:
    duration = stage.end_step - previous_end
    token_budget += (
      duration
      * stage.batch_size
      * stage.seq_len
      * config.loop.gradient_accumulation_steps
    )
    previous_end = stage.end_step

  assert token_budget == 37_048_320


def test_16m_speedrun_uses_automatic_workflow_cadence() -> None:
  experiment = load_experiment_config(FULL_CONFIG)

  assert experiment.eval is not None
  assert experiment.eval.every_steps is None
  assert experiment.eval.every_fraction == 0.01
  assert experiment.eval.min_every_steps == 50
  assert experiment.rollout is not None
  assert experiment.rollout.every_steps is None
  assert experiment.rollout.every_fraction == 0.02
  assert experiment.rollout.min_every_steps == 100
  assert experiment.checkpoint.every_steps is None
  assert experiment.checkpoint.every_fraction == 0.05
  assert experiment.checkpoint.min_every_steps == 200


def test_automatic_workflow_cadence_scales_with_total_steps() -> None:
  experiment = load_experiment_config(FULL_CONFIG)
  config_1k = _config_with_resolved_intervals(experiment.to_train_config())
  config_10k = _config_with_resolved_intervals(
    replace(
      experiment.to_train_config(),
      loop=replace(experiment.loop, steps=10_000),
    )
  )

  assert config_1k.eval is not None
  assert config_1k.rollout is not None
  assert config_1k.eval.every_steps == 50
  assert config_1k.rollout.every_steps == 100
  assert config_1k.checkpoint.every_steps == 200
  assert config_10k.eval is not None
  assert config_10k.rollout is not None
  assert config_10k.eval.every_steps == 100
  assert config_10k.rollout.every_steps == 200
  assert config_10k.checkpoint.every_steps == 500
