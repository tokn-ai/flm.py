import json
from pathlib import Path

import pytest
from flm_train.cli import parse_args
from flm_train.experiment import (
  DataConfig,
  ExperimentConfig,
  ExperimentOverrides,
  FilesSinkConfig,
  ModelConfig,
  OutputConfig,
  RunTrainConfig,
  apply_overrides,
  load_experiment_config,
  parse_experiment_config,
  run_experiment,
)


def test_parse_experiment_config_derives_train_config() -> None:
  config = parse_experiment_config(
    {
      "name": "tiny",
      "seed": 7,
      "device": "cpu",
      "data": {
        "repo_root": "src",
        "encoding_name": "cl100k_base",
        "seq_len": 16,
      },
      "model": {
        "name": "reference",
        "d_model": 32,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 64,
      },
      "optimizer": {
        "learning_rate": 1.0e-3,
        "weight_decay": 0.01,
      },
      "train": {
        "batch_size": 2,
        "steps": 5,
      },
      "output": {
        "run_dir": "runs/tiny",
      },
      "sinks": [
        {
          "kind": "files",
          "metrics_jsonl": "train-metrics.jsonl",
        }
      ],
    }
  )

  train_config = config.to_train_config()

  assert train_config.repo_root == Path("src")
  assert train_config.seq_len == 16
  assert train_config.batch_size == 2
  assert train_config.steps == 5
  assert train_config.d_model == 32
  assert train_config.n_layers == 3
  assert train_config.n_heads == 4
  assert train_config.d_ff == 64
  assert train_config.learning_rate == 1.0e-3
  assert train_config.weight_decay == 0.01
  assert train_config.seed == 7
  assert train_config.device == "cpu"
  assert config.sinks == (
    FilesSinkConfig(metrics_jsonl="train-metrics.jsonl"),
  )


def test_parse_experiment_config_rejects_unknown_keys() -> None:
  with pytest.raises(ValueError, match="unknown experiment config keys"):
    parse_experiment_config({"name": "bad", "typo": True})


def test_load_experiment_config_reads_yaml(tmp_path: Path) -> None:
  config_path = tmp_path / "experiment.yaml"
  config_path.write_text(
    """
name: yaml_test
data:
  seq_len: 12
model:
  d_model: 24
train:
  steps: 2
""",
    encoding="utf-8",
  )

  config = load_experiment_config(config_path)

  assert config.name == "yaml_test"
  assert config.data.seq_len == 12
  assert config.model.d_model == 24
  assert config.train.steps == 2


def test_parse_args_accepts_cli_overrides() -> None:
  args = parse_args(
    [
      "experiments/16m_repo.yaml",
      "--device",
      "cpu",
      "--steps",
      "3",
      "--run-dir",
      "/tmp/run",
      "--seed",
      "99",
    ]
  )

  assert args.config == Path("experiments/16m_repo.yaml")
  assert args.device == "cpu"
  assert args.steps == 3
  assert args.run_dir == Path("/tmp/run")
  assert args.seed == 99


def test_apply_overrides_preserves_unspecified_config() -> None:
  config = ExperimentConfig(
    name="override_test",
    seed=1,
    device="cuda",
    train=RunTrainConfig(batch_size=4, steps=10),
  )

  overridden = apply_overrides(
    config,
    ExperimentOverrides(device="cpu", steps=2, run_dir=Path("/tmp/run")),
  )

  assert overridden.seed == 1
  assert overridden.device == "cpu"
  assert overridden.train.batch_size == 4
  assert overridden.train.steps == 2
  assert overridden.run_dir == Path("/tmp/run")


def test_run_experiment_writes_run_artifacts(tmp_path: Path) -> None:
  repo_root = tmp_path / "repo"
  run_dir = tmp_path / "run"
  repo_root.mkdir()
  (repo_root / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = run_experiment(
    ExperimentConfig(
      name="artifact_test",
      data=DataConfig(repo_root=repo_root, seq_len=8),
      model=ModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      train=RunTrainConfig(batch_size=2, steps=1),
      output=OutputConfig(run_dir=run_dir),
    )
  )

  assert result.file_count == 1
  assert (run_dir / "config.json").is_file()
  assert (run_dir / "config.resolved.yaml").is_file()
  status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
  assert status_payload["status"] == "success"
  metrics_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
  assert len(metrics_lines) == 1
  metrics_payload = json.loads(metrics_lines[0])
  assert metrics_payload["step"] == 1
  assert metrics_payload["train/loss"] > 0
  result_payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
  assert result_payload["file_count"] == 1
  assert len(result_payload["losses"]) == 1


def test_run_experiment_uses_custom_files_sink_paths(tmp_path: Path) -> None:
  repo_root = tmp_path / "repo"
  run_dir = tmp_path / "run"
  sink_dir = tmp_path / "sink"
  repo_root.mkdir()
  (repo_root / "model.py").write_text(
    "\n".join(f"def custom_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  run_experiment(
    ExperimentConfig(
      name="custom_sink_test",
      data=DataConfig(repo_root=repo_root, seq_len=8),
      model=ModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      train=RunTrainConfig(batch_size=2, steps=1),
      output=OutputConfig(run_dir=run_dir),
      sinks=(
        FilesSinkConfig(
          run_dir=sink_dir,
          config_json="cfg.json",
          resolved_config_yaml="cfg.yaml",
          status_json="state.json",
          metrics_jsonl="scalars.jsonl",
          result_json="done.json",
        ),
      ),
    )
  )

  assert not run_dir.exists()
  assert (sink_dir / "cfg.json").is_file()
  assert (sink_dir / "cfg.yaml").is_file()
  assert (sink_dir / "state.json").is_file()
  assert (sink_dir / "scalars.jsonl").is_file()
  assert (sink_dir / "done.json").is_file()
