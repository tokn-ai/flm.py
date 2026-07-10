from pathlib import Path

from flm_train.config import load_experiment_config
from flm_train.models import build_model

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
    token_budget += duration * stage.batch_size * stage.seq_len
    previous_end = stage.end_step

  assert token_budget == 37_048_320
