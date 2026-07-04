import json
import os
import threading
from pathlib import Path

import pytest
from flm_train.cli import parse_args
from flm_train.config import (
  ExperimentConfig,
  ExperimentOverrides,
  FilesSinkConfig,
  MlflowSinkConfig,
  OutputConfig,
  RunConfig,
  SystemMetricsConfig,
  TensorBoardSinkConfig,
  WandbSinkConfig,
  apply_overrides,
  config_to_plain,
  load_experiment_config,
  parse_experiment_config,
)
from flm_train.data import publish_repo_source_dataset
from flm_train.runner import resolve_run_config, run_experiment
from flm_train.secrets import apply_secret_env, load_secret_env
from flm_train.sinks import (
  FilesRunSink,
  MlflowRunSink,
  RunContext,
  TensorBoardRunSink,
  WandbRunSink,
  build_run_sink,
)
from flm_train.system_metrics import SystemMetricsSampler
from flm_train.types import (
  CheckpointConfig,
  DataConfig,
  EvalConfig,
  LoopConfig,
  ReferenceModelConfig,
  RolloutConfig,
  RolloutPromptConfig,
  TrainingResult,
)


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
        "attention_backend": "tilelang",
        "loss_backend": "linear_cross_entropy",
        "loss_chunk_size": 16,
      },
      "optimizer": {
        "kind": "adamw",
        "learning_rate": 1.0e-3,
        "weight_decay": 0.01,
        "max_grad_norm": 0.5,
      },
      "loop": {
        "seed": 7,
        "device": "cpu",
        "dtype": "bfloat16",
        "batch_size": 2,
        "steps": 5,
      },
      "eval": {
        "split": "test",
        "every_steps": 2,
        "max_batches": 3,
      },
      "rollout": {
        "every_steps": 2,
        "max_new_tokens": 8,
        "prompts": [
          {
            "name": "fib",
            "prompt": "def fib(n):",
          }
        ],
      },
      "system_metrics": {
        "enabled": True,
        "every_seconds": 2.5,
      },
      "checkpoint": {
        "enabled": True,
        "every_steps": 2,
        "keep_last": 1,
        "resume": "auto",
      },
      "run": {
        "id": "run-123",
        "name": "tiny brisk-signal",
        "group": "smoke",
      },
      "secrets": {
        "env_file": ".secret",
      },
      "output": {
        "root_dir": "runs",
      },
      "sinks": [
        {
          "kind": "files",
          "metrics_jsonl": "train-metrics.jsonl",
          "system_metrics_jsonl": "system.jsonl",
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
  assert train_config.model.attention_backend == "tilelang"
  assert train_config.model.loss_backend == "linear_cross_entropy"
  assert train_config.model.loss_chunk_size == 16
  assert train_config.optimizer.learning_rate == 1.0e-3
  assert train_config.optimizer.weight_decay == 0.01
  assert train_config.optimizer.max_grad_norm == 0.5
  assert train_config.loop.seed == 7
  assert train_config.loop.device == "cpu"
  assert train_config.loop.dtype == "bfloat16"
  assert train_config.eval == EvalConfig(
    split="test",
    every_steps=2,
    max_batches=3,
  )
  assert train_config.rollout == RolloutConfig(
    every_steps=2,
    max_new_tokens=8,
    prompts=(RolloutPromptConfig(name="fib", prompt="def fib(n):"),),
  )
  assert config.system_metrics == SystemMetricsConfig(
    enabled=True,
    every_seconds=2.5,
  )
  assert config.checkpoint == CheckpointConfig(
    enabled=True,
    every_steps=2,
    keep_last=1,
    resume="auto",
  )
  assert config.run == RunConfig(
    id="run-123",
    name="tiny brisk-signal",
    group="smoke",
  )
  assert config.secrets.env_file == Path(".secret")
  assert config.sinks == (
    FilesSinkConfig(
      metrics_jsonl="train-metrics.jsonl",
      system_metrics_jsonl="system.jsonl",
    ),
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


def test_parse_experiment_config_rejects_output_run_dir() -> None:
  with pytest.raises(ValueError, match="unknown output config keys"):
    parse_experiment_config(
      {
        "name": "bad",
        "output": {
          "run_dir": "runs/bad",
        },
      }
    )


def test_parse_experiment_config_accepts_null_optional_sections() -> None:
  config = parse_experiment_config(
    {
      "name": "resolved",
      "eval": None,
      "rollout": None,
      "sinks": [
        {
          "kind": "files",
          "run_dir": None,
        },
        {
          "kind": "tensorboard",
          "log_dir": None,
        },
        {
          "kind": "wandb",
          "dir": None,
        },
      ],
    }
  )

  assert config.eval is None
  assert config.rollout is None
  assert isinstance(config.sinks[0], FilesSinkConfig)
  assert config.sinks[0].run_dir is None
  assert isinstance(config.sinks[1], TensorBoardSinkConfig)
  assert config.sinks[1].log_dir is None
  assert isinstance(config.sinks[2], WandbSinkConfig)
  assert config.sinks[2].dir is None


def test_resolved_config_generates_run_identity() -> None:
  run = resolve_run_config("identity", RunConfig())

  assert run.id is not None
  assert run.name is not None
  assert run.name.startswith("identity ")
  assert ExperimentConfig(name="identity", run=run).run_dir == (
    Path("runs") / "identity" / run.id
  )


def test_run_dir_uses_explicit_run_id() -> None:
  config = ExperimentConfig(name="identity", run=RunConfig(id="run-123"))

  assert config.run_dir == Path("runs") / "identity" / "run-123"


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
      "--root-dir",
      "/tmp/runs",
      "--seed",
      "99",
    ]
  )

  assert args.config == Path("experiments/16m_repo.yaml")
  assert args.device == "cpu"
  assert args.steps == 3
  assert args.root_dir == Path("/tmp/runs")
  assert args.seed == 99


def test_apply_overrides_preserves_unspecified_config() -> None:
  config = ExperimentConfig(
    name="override_test",
    loop=LoopConfig(seed=1, device="cuda", batch_size=4, steps=10),
  )

  overridden = apply_overrides(
    config,
    ExperimentOverrides(device="cpu", steps=2, root_dir=Path("/tmp/runs")),
  )

  assert overridden.loop.seed == 1
  assert overridden.loop.device == "cpu"
  assert overridden.loop.dtype == "float32"
  assert overridden.loop.batch_size == 4
  assert overridden.loop.steps == 2
  assert overridden.secrets == config.secrets
  assert overridden.run_dir == Path("/tmp/runs") / "override_test"


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
    "attention_backend": "torch",
    "loss_backend": "cross_entropy",
    "loss_chunk_size": 512,
  }


def test_16m_repo_config_uses_sdpa_attention_backend() -> None:
  config = load_experiment_config(Path("experiments/16m_repo.yaml"))

  assert config.data.encoding_name == "unitoken:.cache/tokenizers/repo_8192"
  assert config.model.d_model == 256
  assert config.model.n_layers == 12
  assert config.model.n_heads == 16
  assert config.model.d_ff == 1024
  assert config.model.attention_backend == "torch"
  assert config.model.loss_backend == "cut_cross_entropy"
  assert config.optimizer.max_grad_norm == 1.0
  assert config.loop.dtype == "bfloat16"


def test_100mib_4k_repo_config_uses_benchmarked_shape() -> None:
  config = load_experiment_config(Path("experiments/100mib_4k_repo.yaml"))

  assert config.data.encoding_name == "unitoken:.cache/tokenizers/repo_8192"
  assert config.data.seq_len == 4096
  assert config.model.d_model == 384
  assert config.model.n_layers == 20
  assert config.model.n_heads == 24
  assert config.model.d_ff == 1536
  assert config.model.attention_backend == "torch"
  assert config.model.loss_backend == "cut_cross_entropy"
  assert config.optimizer.max_grad_norm == 1.0
  assert config.loop.dtype == "bfloat16"
  assert config.loop.batch_size == 2
  assert config.eval is not None
  assert config.eval.every_steps == 10
  assert config.eval.max_batches == 4
  assert config.rollout is not None
  assert config.rollout.every_steps == 50
  assert config.checkpoint.every_steps == 1000


def test_parse_experiment_config_accepts_cut_cross_entropy_backend() -> None:
  config = parse_experiment_config(
    {
      "name": "cut_cce",
      "model": {
        "kind": "reference",
        "loss_backend": "cut_cross_entropy",
      },
    }
  )

  assert config.model.loss_backend == "cut_cross_entropy"


def test_run_experiment_writes_run_artifacts(tmp_path: Path) -> None:
  dataset_root = publish_fixture_dataset(tmp_path)
  run_dir = tmp_path / "runs" / "artifact_test" / "run-123"

  result = run_experiment(
    ExperimentConfig(
      name="artifact_test",
      run=RunConfig(id="run-123"),
      data=DataConfig(dataset_root=dataset_root, seq_len=8),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=1),
      output=OutputConfig(root_dir=tmp_path / "runs"),
    )
  )

  assert result.file_count == 1
  assert (run_dir / "config.json").is_file()
  assert (run_dir / "config.resolved.yaml").is_file()
  status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
  assert status_payload["status"] == "success"
  assert status_payload["experiment_name"] == "artifact_test"
  assert status_payload["run_id"]
  assert status_payload["run_name"].startswith("artifact_test ")
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
  system_metrics_lines = (
    (run_dir / "system_metrics.jsonl").read_text(encoding="utf-8").splitlines()
  )
  assert len(system_metrics_lines) >= 1
  system_metrics_payload = json.loads(system_metrics_lines[0])
  assert "time" in system_metrics_payload
  assert "process" in system_metrics_payload
  assert "gpus" in system_metrics_payload
  result_payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
  assert result_payload["file_count"] == 1
  assert len(result_payload["losses"]) == 1


def test_run_experiment_resolves_latest_dataset_version(tmp_path: Path) -> None:
  repo_root = tmp_path / "repo"
  dataset_root = tmp_path / "datasets" / "repo_sources"
  run_dir = tmp_path / "runs" / "resolved_dataset_test" / "run-123"
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
      run=RunConfig(id="run-123"),
      data=DataConfig(
        kind="token_dataset",
        dataset_root=dataset_root,
        version="latest",
        split="train",
        seq_len=8,
      ),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=1),
      output=OutputConfig(root_dir=tmp_path / "runs"),
    )
  )

  resolved = (run_dir / "config.resolved.yaml").read_text(encoding="utf-8")
  assert "version: latest" in resolved
  assert "split: train" in resolved
  assert f"resolved_version: {published.version}" in resolved


def test_run_experiment_logs_eval_and_rollout(tmp_path: Path) -> None:
  dataset_root = publish_split_fixture_dataset(tmp_path)
  run_dir = tmp_path / "runs" / "eval_rollout_test" / "run-123"

  run_experiment(
    ExperimentConfig(
      name="eval_rollout_test",
      run=RunConfig(id="run-123"),
      data=DataConfig(dataset_root=dataset_root, seq_len=8),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=2),
      eval=EvalConfig(split="test", every_steps=1, max_batches=1),
      rollout=RolloutConfig(
        every_steps=2,
        max_new_tokens=2,
        prompts=(RolloutPromptConfig(name="fib", prompt="def fib(n):"),),
      ),
      output=OutputConfig(root_dir=tmp_path / "runs"),
    )
  )

  metrics_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
  eval_metrics = [
    json.loads(line) for line in metrics_lines if "eval/test_loss" in line
  ]
  assert [metrics["step"] for metrics in eval_metrics] == [1, 2]
  assert all(metrics["eval/test_loss"] > 0 for metrics in eval_metrics)
  assert all("eval/test_perplexity" not in metrics for metrics in eval_metrics)

  rollout_path = run_dir / "rollouts" / "step-00000002.json"
  rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
  assert rollout["step"] == 2
  assert rollout["samples"][0]["name"] == "fib"
  assert rollout["samples"][0]["prompt"] == "def fib(n):"
  assert rollout["samples"][0]["prompt_tokens"]
  assert len(rollout["samples"][0]["tokens"]) == 2
  assert "completion" not in rollout["samples"][0]
  assert len(rollout["samples"][0]["token_texts"]) == 2
  assert len(rollout["samples"][0]["log_probs"]) == 2
  assert len(rollout["samples"][0]["entropy"]) == 2
  assert len(rollout["samples"][0]["top_tokens"]) == 2
  assert len(rollout["samples"][0]["top_token_texts"]) == 2
  assert len(rollout["samples"][0]["top_log_probs"]) == 2
  assert all(isinstance(token, int) for token in rollout["samples"][0]["tokens"])
  assert all(len(tokens) == 10 for tokens in rollout["samples"][0]["top_tokens"])
  assert all(len(texts) == 10 for texts in rollout["samples"][0]["top_token_texts"])
  assert all(
    len(log_probs) == 10 for log_probs in rollout["samples"][0]["top_log_probs"]
  )
  artifacts = (run_dir / "artifacts.jsonl").read_text(encoding="utf-8")
  assert "rollouts/step-00000002.json" in artifacts


def test_run_experiment_writes_checkpoints(tmp_path: Path) -> None:
  dataset_root = publish_fixture_dataset(tmp_path)
  run_dir = tmp_path / "runs" / "checkpoint_test" / "run-123"

  run_experiment(
    ExperimentConfig(
      name="checkpoint_test",
      run=RunConfig(id="run-123"),
      data=DataConfig(dataset_root=dataset_root, seq_len=8),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=2),
      checkpoint=CheckpointConfig(enabled=True, every_steps=1, keep_last=1),
      output=OutputConfig(root_dir=tmp_path / "runs"),
    )
  )

  checkpoint_dir = run_dir / "checkpoints"
  latest = checkpoint_dir / "latest"
  assert latest.read_text(encoding="utf-8").strip() == "step-00000002"
  assert not (checkpoint_dir / "step-00000001").exists()
  checkpoint = checkpoint_dir / "step-00000002"
  assert (checkpoint / "model.npz").is_file()
  assert (checkpoint / "optimizer.npz").is_file()
  assert (checkpoint / "trainer_state.json").is_file()
  manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
  assert manifest["format"] == "flm-checkpoint-v2"
  assert manifest["step"] == 2
  model_state = json.loads(
    (checkpoint / "model_state.json").read_text(encoding="utf-8")
  )
  first_tensor = next(
    value["__tensor__"]
    for value in model_state.values()
    if isinstance(value, dict) and "__tensor__" in value
  )
  assert first_tensor["name"]
  assert first_tensor["shape"]
  assert first_tensor["dtype"].startswith("torch.")
  assert first_tensor["device"] == "cpu"
  artifacts = (run_dir / "artifacts.jsonl").read_text(encoding="utf-8")
  assert "checkpoints/step-00000002" in artifacts


def test_run_experiment_uses_custom_files_sink_paths(tmp_path: Path) -> None:
  dataset_root = publish_fixture_dataset(tmp_path)
  root_dir = tmp_path / "runs"
  sink_dir = tmp_path / "sink"

  run_experiment(
    ExperimentConfig(
      name="custom_sink_test",
      run=RunConfig(id="run-123"),
      data=DataConfig(dataset_root=dataset_root, seq_len=8),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=1),
      output=OutputConfig(root_dir=root_dir),
      sinks=(
        FilesSinkConfig(
          run_dir=sink_dir,
          config_json="cfg.json",
          resolved_config_yaml="cfg.yaml",
          status_json="state.json",
          metrics_jsonl="scalars.jsonl",
          system_metrics_jsonl="system.jsonl",
          result_json="done.json",
        ),
      ),
    )
  )

  assert not root_dir.exists()
  assert (sink_dir / "cfg.json").is_file()
  assert (sink_dir / "cfg.yaml").is_file()
  assert (sink_dir / "state.json").is_file()
  assert (sink_dir / "scalars.jsonl").is_file()
  assert (sink_dir / "system.jsonl").is_file()
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


def test_files_sink_logs_system_metrics(tmp_path: Path) -> None:
  sink = FilesRunSink(FilesSinkConfig(system_metrics_jsonl="system.jsonl"))

  sink.start_run(RunContext(run_dir=tmp_path), ExperimentConfig(name="files"))
  sink.log_system_metrics(
    {
      "process": {"pid": 123, "max_rss_bytes": 456},
      "gpus": [
        {
          "source": "nvidia-smi",
          "index": 0,
          "utilization_pct": 12.5,
        }
      ],
    }
  )

  payload = json.loads((tmp_path / "system.jsonl").read_text(encoding="utf-8"))
  assert "time" in payload
  assert payload["process"]["pid"] == 123
  assert payload["gpus"][0]["utilization_pct"] == 12.5


def test_system_metrics_sampler_emits_immediately() -> None:
  emissions = []
  emitted = threading.Event()

  def emit(metrics) -> None:
    emissions.append(metrics)
    emitted.set()

  sampler = SystemMetricsSampler(
    every_seconds=60.0,
    collect=lambda: {"process": {"pid": 1}, "gpus": []},
    emit=emit,
  )

  sampler.start()
  assert emitted.wait(timeout=1.0)
  sampler.stop()

  assert emissions == [{"process": {"pid": 1}, "gpus": []}]


def test_tensorboard_sink_logs_scalars_and_text(tmp_path: Path) -> None:
  writer = FakeSummaryWriter()
  sink = TensorBoardRunSink(TensorBoardSinkConfig(), writer=writer)

  sink.start_run(
    RunContext(run_dir=tmp_path),
    ExperimentConfig(
      name="tb",
      run=RunConfig(id="run-123", name="tb brisk-signal", group="smoke"),
    ),
  )
  sink.log_metrics({"train/loss": 1.5, "phase": "train"}, step=2)
  sink.log_system_metrics(
    {
      "process": {"max_rss_bytes": 1024},
      "gpus": [{"index": 0, "name": "gpu", "utilization_pct": 25.0}],
    }
  )
  sink.log_artifact(tmp_path / "checkpoint.pt")
  sink.finish_run(TrainingResult(losses=[1.5], token_count=10, file_count=1))
  sink.close()

  assert ("train/loss", 1.5, 2) in writer.scalars
  assert ("phase", "train", 2) in writer.texts
  assert ("experiment/name", "tb", 0) in writer.texts
  assert ("run/id", "run-123", 0) in writer.texts
  assert ("run/name", "tb brisk-signal", 0) in writer.texts
  assert ("run/group", "smoke", 0) in writer.texts
  assert ("system/process/max_rss_bytes", 1024, 0) in writer.scalars
  assert ("system/gpus/0/utilization_pct", 25.0, 0) in writer.scalars
  assert ("system/gpus/0/name", "gpu", 0) in writer.texts
  assert writer.closed


def test_mlflow_sink_logs_run_data(tmp_path: Path) -> None:
  client = FakeMlflow()
  sink = MlflowRunSink(
    MlflowSinkConfig(
      tracking_uri="file:mlruns",
      nested=True,
    ),
    client=client,
  )

  sink.start_run(
    RunContext(run_dir=tmp_path),
    ExperimentConfig(
      name="mlflow",
      run=RunConfig(id="run-123", name="mlflow brisk-signal", group="smoke"),
    ),
  )
  sink.log_metrics({"train/loss": 1.25, "phase": "train"}, step=3)
  sink.log_system_metrics(
    {
      "process": {"max_rss_bytes": 2048},
      "gpus": [{"index": 0, "name": "gpu", "utilization_pct": 50.0}],
    }
  )
  sink.log_artifact(tmp_path / "artifact.txt", name="artifacts")
  sink.finish_run(TrainingResult(losses=[1.25], token_count=10, file_count=1))

  assert client.tracking_uri == "file:mlruns"
  assert client.experiment_name == "mlflow"
  assert client.started == {"run_name": "mlflow brisk-signal", "nested": True}
  assert client.tags[0] == {
    "flm.experiment_name": "mlflow",
    "flm.run_id": "run-123",
    "flm.run_name": "mlflow brisk-signal",
    "flm.run_group": "smoke",
  }
  assert client.metrics == [
    ({"train/loss": 1.25}, 3),
    (
      {
        "system/process/max_rss_bytes": 2048.0,
        "system/gpus/0/index": 0.0,
        "system/gpus/0/utilization_pct": 50.0,
      },
      0,
    ),
  ]
  assert client.artifacts == [(str(tmp_path / "artifact.txt"), "artifacts")]
  assert client.ended == ["FINISHED"]


def test_wandb_sink_logs_run_data(tmp_path: Path) -> None:
  module = FakeWandb()
  sink = WandbRunSink(
    WandbSinkConfig(
      project="project",
      entity="entity",
      mode="offline",
      tags=("tag-a",),
      job_type="job",
    ),
    module=module,
  )

  sink.start_run(
    RunContext(run_dir=tmp_path),
    ExperimentConfig(
      name="wandb",
      run=RunConfig(id="run-123", name="wandb brisk-signal", group="smoke"),
    ),
  )
  sink.log_metrics({"train/loss": 2.0}, step=4)
  sink.log_system_metrics(
    {
      "process": {"max_rss_bytes": 4096},
      "gpus": [{"index": 0, "name": "gpu", "utilization_pct": 75.0}],
    }
  )
  sink.log_artifact(tmp_path / "artifact.txt")
  sink.finish_run(TrainingResult(losses=[2.0], token_count=10, file_count=1))

  assert module.init_kwargs["project"] == "project"
  assert module.init_kwargs["entity"] == "entity"
  assert module.init_kwargs["id"] == "run-123"
  assert module.init_kwargs["name"] == "wandb brisk-signal"
  assert module.init_kwargs["group"] == "smoke"
  assert module.logs[-3] == ({"train/loss": 2.0}, 4)
  assert module.logs[-2] == (
    {
      "system/process/max_rss_bytes": 4096,
      "system/gpus/0/index": 0,
      "system/gpus/0/name": "gpu",
      "system/gpus/0/utilization_pct": 75.0,
      "system/sample": 0,
    },
    None,
  )
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


def publish_split_fixture_dataset(tmp_path: Path) -> Path:
  repo_root = tmp_path / "repo"
  dataset_root = tmp_path / "datasets" / "repo_sources"
  repo_root.mkdir()
  for index in range(80):
    (repo_root / f"model_{index}.py").write_text(
      "\n".join(f"def f_{index}_{i}(): return {i}" for i in range(20)),
      encoding="utf-8",
    )
  published = publish_repo_source_dataset(
    repo_root=repo_root,
    dataset_root=dataset_root,
    train_ratio=0.5,
    val_ratio=0.0,
    test_ratio=0.5,
  )
  assert published.splits["train"]["token_count"] > 8
  assert published.splits["test"]["token_count"] > 8
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
