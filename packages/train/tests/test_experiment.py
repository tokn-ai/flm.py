import json
import os
from pathlib import Path

import pytest
from flm_train.cli import parse_args
from flm_train.config import (
  ExperimentConfig,
  ExperimentOverrides,
  FilesSinkConfig,
  MlflowSinkConfig,
  OutputConfig,
  TensorBoardSinkConfig,
  WandbSinkConfig,
  apply_overrides,
  config_to_plain,
  load_experiment_config,
  parse_experiment_config,
)
from flm_train.data import publish_repo_source_dataset
from flm_train.runner import run_experiment
from flm_train.secrets import apply_secret_env, load_secret_env
from flm_train.sinks import (
  MlflowRunSink,
  RunContext,
  TensorBoardRunSink,
  WandbRunSink,
  build_run_sink,
)
from flm_train.types import DataConfig, LoopConfig, ReferenceModelConfig, TrainingResult


def test_parse_experiment_config_derives_train_config() -> None:
  config = parse_experiment_config(
    {
      "name": "tiny",
      "data": {
        "dataset_root": "datasets/src",
        "encoding_name": "cl100k_base",
        "seq_len": 16,
      },
      "model": {
        "kind": "reference",
        "d_model": 32,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 64,
      },
      "optimizer": {
        "kind": "adamw",
        "learning_rate": 1.0e-3,
        "weight_decay": 0.01,
      },
      "loop": {
        "seed": 7,
        "device": "cpu",
        "batch_size": 2,
        "steps": 5,
      },
      "secrets": {
        "env_file": ".secret",
      },
      "output": {
        "run_dir": "runs/tiny",
      },
      "sinks": [
        {
          "kind": "files",
          "metrics_jsonl": "train-metrics.jsonl",
        },
        {
          "kind": "tensorboard",
          "log_dir": "tb",
          "flush_secs": 3,
        },
        {
          "kind": "mlflow",
          "tracking_uri": "file:mlruns",
          "experiment_name": "flm-test",
          "run_name": "run-a",
          "nested": True,
        },
        {
          "kind": "wandb",
          "project": "flm",
          "entity": "team",
          "name": "wandb-run",
          "mode": "offline",
          "dir": "wandb",
          "tags": ["smoke", "train"],
          "group": "group-a",
          "job_type": "pretrain",
        },
      ],
    }
  )

  train_config = config.to_train_config()

  assert train_config.data.kind == "token_dataset"
  assert train_config.data.dataset_root == Path("datasets/src")
  assert train_config.data.seq_len == 16
  assert train_config.loop.batch_size == 2
  assert train_config.loop.steps == 5
  assert train_config.model.d_model == 32
  assert train_config.model.n_layers == 3
  assert train_config.model.n_heads == 4
  assert train_config.model.d_ff == 64
  assert train_config.optimizer.learning_rate == 1.0e-3
  assert train_config.optimizer.weight_decay == 0.01
  assert train_config.loop.seed == 7
  assert train_config.loop.device == "cpu"
  assert config.secrets.env_file == Path(".secret")
  assert config.sinks == (
    FilesSinkConfig(metrics_jsonl="train-metrics.jsonl"),
    TensorBoardSinkConfig(log_dir=Path("tb"), flush_secs=3),
    MlflowSinkConfig(
      tracking_uri="file:mlruns",
      experiment_name="flm-test",
      run_name="run-a",
      nested=True,
    ),
    WandbSinkConfig(
      project="flm",
      entity="team",
      name="wandb-run",
      mode="offline",
      dir=Path("wandb"),
      tags=("smoke", "train"),
      group="group-a",
      job_type="pretrain",
    ),
  )


def test_parse_experiment_config_rejects_unknown_keys() -> None:
  with pytest.raises(ValueError, match="unknown experiment config keys"):
    parse_experiment_config({"name": "bad", "typo": True})


def test_parse_experiment_config_rejects_live_repo_data() -> None:
  with pytest.raises(ValueError, match="unsupported data.kind: repo_sources"):
    parse_experiment_config(
      {
        "name": "bad",
        "data": {
          "kind": "repo_sources",
        },
      }
    )


def test_parse_experiment_config_rejects_unknown_data_keys() -> None:
  with pytest.raises(ValueError, match="unknown data config keys"):
    parse_experiment_config(
      {
        "name": "bad",
        "data": {
          "repo_root": ".",
        },
      }
    )


def test_parse_experiment_config_rejects_unknown_split() -> None:
  with pytest.raises(ValueError, match="unsupported data.split"):
    parse_experiment_config(
      {
        "name": "bad",
        "data": {
          "split": "valid",
        },
      }
    )


def test_load_experiment_config_reads_yaml(tmp_path: Path) -> None:
  config_path = tmp_path / "experiment.yaml"
  config_path.write_text(
    """
name: yaml_test
data:
  seq_len: 12
model:
  d_model: 24
loop:
  steps: 2
""",
    encoding="utf-8",
  )

  config = load_experiment_config(config_path)

  assert config.name == "yaml_test"
  assert config.data.seq_len == 12
  assert config.model.d_model == 24
  assert config.loop.steps == 2


def test_parse_experiment_config_reads_token_dataset_config() -> None:
  config = parse_experiment_config(
    {
      "name": "prepared",
      "data": {
        "kind": "token_dataset",
        "dataset_root": ".cache/data/repo_sources",
        "version": "latest",
        "split": "val",
        "encoding_name": "cl100k_base",
        "seq_len": 512,
      },
    }
  )

  assert config.data.kind == "token_dataset"
  assert config.data.dataset_root == Path(".cache/data/repo_sources")
  assert config.data.version == "latest"
  assert config.data.split == "val"
  assert config.data.seq_len == 512


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
    loop=LoopConfig(seed=1, device="cuda", batch_size=4, steps=10),
  )

  overridden = apply_overrides(
    config,
    ExperimentOverrides(device="cpu", steps=2, run_dir=Path("/tmp/run")),
  )

  assert overridden.loop.seed == 1
  assert overridden.loop.device == "cpu"
  assert overridden.loop.batch_size == 4
  assert overridden.loop.steps == 2
  assert overridden.secrets == config.secrets
  assert overridden.run_dir == Path("/tmp/run")


def test_secret_env_file_loads_dotenv_values(tmp_path: Path) -> None:
  secret_path = tmp_path / ".secret"
  secret_path.write_text(
    """
# local secrets
WANDB_API_KEY=abc123
MLFLOW_TRACKING_TOKEN="token value"
EMPTY=
""",
    encoding="utf-8",
  )

  assert load_secret_env(secret_path) == {
    "WANDB_API_KEY": "abc123",
    "MLFLOW_TRACKING_TOKEN": "token value",
    "EMPTY": "",
  }


def test_apply_secret_env_does_not_overwrite_existing_env(monkeypatch) -> None:
  monkeypatch.setenv("WANDB_API_KEY", "from-env")

  apply_secret_env({"WANDB_API_KEY": "from-file", "MLFLOW_TRACKING_TOKEN": "token"})

  assert os.environ["WANDB_API_KEY"] == "from-env"
  assert os.environ["MLFLOW_TRACKING_TOKEN"] == "token"


def test_config_plain_includes_secret_path_only() -> None:
  config = parse_experiment_config(
    {
      "name": "secret_path_test",
      "secrets": {"env_file": ".secret"},
    }
  )
  plain = config_to_plain(config)

  assert plain["secrets"] == {"env_file": ".secret"}
  assert "WANDB_API_KEY" not in json.dumps(plain)


def test_reference_model_config_excludes_other_model_fields() -> None:
  config = parse_experiment_config(
    {
      "name": "reference_only",
      "model": {
        "kind": "reference",
        "d_model": 32,
        "n_layers": 3,
        "n_heads": 4,
        "d_ff": 64,
      },
    }
  )
  plain = config_to_plain(config)

  assert plain["model"] == {
    "kind": "reference",
    "d_model": 32,
    "n_layers": 3,
    "n_heads": 4,
    "d_ff": 64,
  }


def test_run_experiment_writes_run_artifacts(tmp_path: Path) -> None:
  dataset_root = publish_fixture_dataset(tmp_path)
  run_dir = tmp_path / "run"

  result = run_experiment(
    ExperimentConfig(
      name="artifact_test",
      data=DataConfig(dataset_root=dataset_root, seq_len=8),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=1),
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
  assert metrics_payload["train/lr"] == 3e-4
  assert metrics_payload["train/tokens"] == 16
  assert metrics_payload["train/tokens_seen"] == 16
  assert metrics_payload["train/tokens_per_sec"] > 0
  assert metrics_payload["system/step_time_sec"] > 0
  result_payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
  assert result_payload["file_count"] == 1
  assert len(result_payload["losses"]) == 1


def test_run_experiment_resolves_latest_dataset_version(tmp_path: Path) -> None:
  repo_root = tmp_path / "repo"
  dataset_root = tmp_path / "datasets" / "repo_sources"
  run_dir = tmp_path / "run"
  repo_root.mkdir()
  (repo_root / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )
  published = publish_repo_source_dataset(
    repo_root=repo_root,
    dataset_root=dataset_root,
  )

  run_experiment(
    ExperimentConfig(
      name="resolved_dataset_test",
      data=DataConfig(
        kind="token_dataset",
        dataset_root=dataset_root,
        version="latest",
        split="train",
        seq_len=8,
      ),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=1),
      output=OutputConfig(run_dir=run_dir),
    )
  )

  resolved = (run_dir / "config.resolved.yaml").read_text(encoding="utf-8")
  assert "version: latest" in resolved
  assert "split: train" in resolved
  assert f"resolved_version: {published.version}" in resolved


def test_run_experiment_uses_custom_files_sink_paths(tmp_path: Path) -> None:
  dataset_root = publish_fixture_dataset(tmp_path)
  run_dir = tmp_path / "run"
  sink_dir = tmp_path / "sink"

  run_experiment(
    ExperimentConfig(
      name="custom_sink_test",
      data=DataConfig(dataset_root=dataset_root, seq_len=8),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=1),
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


def test_build_run_sink_builds_all_sink_kinds() -> None:
  sink = build_run_sink(
    ExperimentConfig(
      name="all_sinks",
      sinks=(
        FilesSinkConfig(),
        TensorBoardSinkConfig(),
        MlflowSinkConfig(),
        WandbSinkConfig(),
      ),
    )
  )

  assert len(sink.sinks) == 4


def test_tensorboard_sink_logs_scalars_and_text(tmp_path: Path) -> None:
  writer = FakeSummaryWriter()
  sink = TensorBoardRunSink(TensorBoardSinkConfig(), writer=writer)

  sink.start_run(RunContext(run_dir=tmp_path), ExperimentConfig(name="tb"))
  sink.log_metrics({"train/loss": 1.5, "phase": "train"}, step=2)
  sink.log_artifact(tmp_path / "checkpoint.pt")
  sink.finish_run(TrainingResult(losses=[1.5], token_count=10, file_count=1))
  sink.close()

  assert ("train/loss", 1.5, 2) in writer.scalars
  assert ("phase", "train", 2) in writer.texts
  assert writer.closed


def test_mlflow_sink_logs_run_data(tmp_path: Path) -> None:
  client = FakeMlflow()
  sink = MlflowRunSink(
    MlflowSinkConfig(
      tracking_uri="file:mlruns",
      experiment_name="exp",
      run_name="run",
      nested=True,
    ),
    client=client,
  )

  sink.start_run(RunContext(run_dir=tmp_path), ExperimentConfig(name="mlflow"))
  sink.log_metrics({"train/loss": 1.25, "phase": "train"}, step=3)
  sink.log_artifact(tmp_path / "artifact.txt", name="artifacts")
  sink.finish_run(TrainingResult(losses=[1.25], token_count=10, file_count=1))

  assert client.tracking_uri == "file:mlruns"
  assert client.experiment_name == "exp"
  assert client.started == {"run_name": "run", "nested": True}
  assert client.metrics == [({"train/loss": 1.25}, 3)]
  assert client.artifacts == [(str(tmp_path / "artifact.txt"), "artifacts")]
  assert client.ended == ["FINISHED"]


def test_wandb_sink_logs_run_data(tmp_path: Path) -> None:
  module = FakeWandb()
  sink = WandbRunSink(
    WandbSinkConfig(
      project="project",
      entity="entity",
      name="run",
      mode="offline",
      tags=("tag-a",),
      group="group",
      job_type="job",
    ),
    module=module,
  )

  sink.start_run(RunContext(run_dir=tmp_path), ExperimentConfig(name="wandb"))
  sink.log_metrics({"train/loss": 2.0}, step=4)
  sink.log_artifact(tmp_path / "artifact.txt")
  sink.finish_run(TrainingResult(losses=[2.0], token_count=10, file_count=1))

  assert module.init_kwargs["project"] == "project"
  assert module.init_kwargs["entity"] == "entity"
  assert module.init_kwargs["name"] == "run"
  assert module.logs[-2] == ({"train/loss": 2.0}, 4)
  assert module.artifacts[0].files == [str(tmp_path / "artifact.txt")]
  assert module.run.finished


def publish_fixture_dataset(tmp_path: Path) -> Path:
  repo_root = tmp_path / "repo"
  dataset_root = tmp_path / "datasets" / "repo_sources"
  repo_root.mkdir()
  (repo_root / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )
  publish_repo_source_dataset(
    repo_root=repo_root,
    dataset_root=dataset_root,
    train_ratio=1.0,
    val_ratio=0.0,
    test_ratio=0.0,
  )
  return dataset_root


class FakeSummaryWriter:
  def __init__(self) -> None:
    self.scalars = []
    self.texts = []
    self.closed = False

  def add_scalar(self, name, value, step) -> None:
    self.scalars.append((name, value, step))

  def add_text(self, name, value, step) -> None:
    self.texts.append((name, value, step))

  def close(self) -> None:
    self.closed = True


class FakeMlflow:
  def __init__(self) -> None:
    self.tracking_uri = None
    self.experiment_name = None
    self.started = None
    self.params = []
    self.tags = []
    self.metrics = []
    self.artifacts = []
    self.dicts = []
    self.ended = []

  def set_tracking_uri(self, uri) -> None:
    self.tracking_uri = uri

  def set_experiment(self, name) -> None:
    self.experiment_name = name

  def start_run(self, *, run_name, nested) -> None:
    self.started = {"run_name": run_name, "nested": nested}

  def log_param(self, name, value) -> None:
    self.params.append((name, value))

  def set_tags(self, tags) -> None:
    self.tags.append(tags)

  def log_metrics(self, metrics, *, step) -> None:
    self.metrics.append((metrics, step))

  def log_artifact(self, path, artifact_path=None) -> None:
    self.artifacts.append((path, artifact_path))

  def log_dict(self, payload, path) -> None:
    self.dicts.append((payload, path))

  def end_run(self, status=None) -> None:
    self.ended.append(status)


class FakeWandbRun:
  def __init__(self) -> None:
    self.config = FakeWandbConfig()
    self.summary = {}
    self.finished = False

  def finish(self) -> None:
    self.finished = True


class FakeWandbConfig(dict):
  def update(self, values, allow_val_change=False) -> None:
    del allow_val_change
    super().update(values)


class FakeWandbArtifact:
  def __init__(self, name, type) -> None:
    self.name = name
    self.type = type
    self.files = []

  def add_file(self, path) -> None:
    self.files.append(path)


class FakeWandb:
  Artifact = FakeWandbArtifact

  def __init__(self) -> None:
    self.run = FakeWandbRun()
    self.init_kwargs = None
    self.logs = []
    self.artifacts = []

  def init(self, **kwargs):
    self.init_kwargs = kwargs
    return self.run

  def log(self, payload, step=None) -> None:
    self.logs.append((payload, step))

  def log_artifact(self, artifact) -> None:
    self.artifacts.append(artifact)
