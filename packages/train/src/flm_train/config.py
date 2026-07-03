"""Experiment configuration types and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from flm_train.types import (
  DataConfig,
  DeepSeekV4ModelConfig,
  DSTinyModelConfig,
  LoopConfig,
  ModelConfig,
  OptimizerConfig,
  ReferenceModelConfig,
  TrainConfig,
)


@dataclass(frozen=True)
class SecretsConfig:
  env_file: Path | None = Path(".secret")


@dataclass(frozen=True)
class OutputConfig:
  run_dir: Path | None = None


@dataclass(frozen=True)
class FilesSinkConfig:
  kind: Literal["files"] = "files"
  run_dir: Path | None = None
  config_json: str = "config.json"
  resolved_config_yaml: str = "config.resolved.yaml"
  status_json: str = "status.json"
  metrics_jsonl: str = "metrics.jsonl"
  result_json: str = "result.json"


@dataclass(frozen=True)
class TensorBoardSinkConfig:
  kind: Literal["tensorboard"] = "tensorboard"
  log_dir: Path | None = None
  flush_secs: int = 10


@dataclass(frozen=True)
class MlflowSinkConfig:
  kind: Literal["mlflow"] = "mlflow"
  tracking_uri: str | None = None
  experiment_name: str = "flm"
  run_name: str | None = None
  nested: bool = False


@dataclass(frozen=True)
class WandbSinkConfig:
  kind: Literal["wandb"] = "wandb"
  project: str = "flm"
  entity: str | None = None
  name: str | None = None
  mode: str | None = None
  dir: Path | None = None
  tags: tuple[str, ...] = ()
  group: str | None = None
  job_type: str | None = None


SinkConfig = (
  FilesSinkConfig | TensorBoardSinkConfig | MlflowSinkConfig | WandbSinkConfig
)


@dataclass(frozen=True)
class ExperimentConfig:
  name: str
  data: DataConfig = field(default_factory=DataConfig)
  model: ModelConfig = field(default_factory=ReferenceModelConfig)
  optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
  loop: LoopConfig = field(default_factory=LoopConfig)
  secrets: SecretsConfig = field(default_factory=SecretsConfig)
  output: OutputConfig = field(default_factory=OutputConfig)
  sinks: tuple[SinkConfig, ...] = field(default_factory=tuple)

  @property
  def run_dir(self) -> Path:
    if self.output.run_dir is not None:
      return self.output.run_dir
    return Path("runs") / self.name

  def to_train_config(self) -> TrainConfig:
    if self.data.kind != "repo_sources":
      raise ValueError(f"unsupported data.kind: {self.data.kind}")
    if self.optimizer.kind != "adamw":
      raise ValueError(f"unsupported optimizer.kind: {self.optimizer.kind}")
    return TrainConfig(
      data=self.data,
      model=self.model,
      optimizer=self.optimizer,
      loop=self.loop,
    )


@dataclass(frozen=True)
class ExperimentOverrides:
  device: str | None = None
  steps: int | None = None
  run_dir: Path | None = None
  seed: int | None = None


def load_experiment_config(path: Path) -> ExperimentConfig:
  raw = yaml.safe_load(path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError("experiment config must be a YAML mapping")
  return parse_experiment_config(raw)


def parse_experiment_config(raw: dict[str, Any]) -> ExperimentConfig:
  allowed = {
    "name",
    "data",
    "model",
    "optimizer",
    "loop",
    "secrets",
    "output",
    "sinks",
  }
  unknown = set(raw) - allowed
  if unknown:
    raise ValueError(f"unknown experiment config keys: {sorted(unknown)}")
  if "name" not in raw:
    raise ValueError("experiment config requires 'name'")

  data = _section(raw, "data")
  model = _section(raw, "model")
  optimizer = _section(raw, "optimizer")
  loop = _section(raw, "loop")
  secrets = _section(raw, "secrets")
  output = _section(raw, "output")

  return ExperimentConfig(
    name=str(raw["name"]),
    data=DataConfig(
      kind=data.get("kind", "repo_sources"),
      repo_root=Path(data.get("repo_root", ".")),
      encoding_name=str(data.get("encoding_name", "cl100k_base")),
      seq_len=int(data.get("seq_len", 128)),
      cache_dir=_optional_path(data.get("cache_dir", ".cache/data")),
    ),
    model=_parse_model(model),
    optimizer=OptimizerConfig(
      kind=optimizer.get("kind", "adamw"),
      learning_rate=float(optimizer.get("learning_rate", 3e-4)),
      weight_decay=float(optimizer.get("weight_decay", 0.1)),
    ),
    loop=LoopConfig(
      seed=int(loop.get("seed", 42)),
      device=str(loop.get("device", "cpu")),
      batch_size=int(loop.get("batch_size", 8)),
      steps=int(loop.get("steps", 10)),
    ),
    secrets=SecretsConfig(
      env_file=_optional_path(secrets.get("env_file", ".secret")),
    ),
    output=OutputConfig(
      run_dir=Path(output["run_dir"]) if "run_dir" in output else None,
    ),
    sinks=_parse_sinks(raw.get("sinks")),
  )


def apply_overrides(
  config: ExperimentConfig,
  overrides: ExperimentOverrides,
) -> ExperimentConfig:
  return ExperimentConfig(
    name=config.name,
    data=config.data,
    model=config.model,
    optimizer=config.optimizer,
    loop=LoopConfig(
      seed=config.loop.seed if overrides.seed is None else overrides.seed,
      device=config.loop.device if overrides.device is None else overrides.device,
      batch_size=config.loop.batch_size,
      steps=config.loop.steps if overrides.steps is None else overrides.steps,
    ),
    secrets=config.secrets,
    output=config.output
    if overrides.run_dir is None
    else OutputConfig(run_dir=overrides.run_dir),
    sinks=config.sinks,
  )


def config_to_plain(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, tuple):
    return [config_to_plain(item) for item in value]
  if hasattr(value, "__dataclass_fields__"):
    return {key: config_to_plain(item) for key, item in asdict(value).items()}
  if isinstance(value, dict):
    return {key: config_to_plain(item) for key, item in value.items()}
  if isinstance(value, list):
    return [config_to_plain(item) for item in value]
  return value


def write_yaml(path: Path, value: Any) -> None:
  path.write_text(
    yaml.safe_dump(value, sort_keys=False),
    encoding="utf-8",
  )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
  value = raw.get(name, {})
  if not isinstance(value, dict):
    raise ValueError(f"experiment config section '{name}' must be a mapping")
  return value


def _optional_int(value: Any) -> int | None:
  if value is None:
    return None
  return int(value)


def _optional_path(value: Any) -> Path | None:
  if value is None:
    return None
  return Path(value)


def _parse_model(value: dict[str, Any]) -> ModelConfig:
  kind = value.get("kind", "reference")
  if kind == "reference":
    return ReferenceModelConfig(
      d_model=int(value.get("d_model", 128)),
      n_layers=int(value.get("n_layers", 2)),
      n_heads=int(value.get("n_heads", 4)),
      d_ff=_optional_int(value.get("d_ff")),
    )
  if kind == "ds_tiny":
    return DSTinyModelConfig(
      d_model=int(value.get("d_model", 128)),
      n_layers=int(value.get("n_layers", 2)),
      n_heads=int(value.get("n_heads", 4)),
      d_ff=_optional_int(value.get("d_ff")),
      q_lora_rank=_optional_int(value.get("q_lora_rank")),
      kv_lora_rank=int(value.get("kv_lora_rank", 64)),
      qk_nope_head_dim=int(value.get("qk_nope_head_dim", 16)),
      qk_rope_head_dim=int(value.get("qk_rope_head_dim", 16)),
      v_head_dim=int(value.get("v_head_dim", 32)),
    )
  if kind == "deepseek_v4":
    return DeepSeekV4ModelConfig(
      d_model=int(value.get("d_model", 128)),
      n_layers=int(value.get("n_layers", 2)),
      n_heads=int(value.get("n_heads", 4)),
      head_dim=_optional_int(value.get("head_dim")),
      d_ff=_optional_int(value.get("d_ff")),
      q_lora_rank=_optional_int(value.get("q_lora_rank")),
      kv_lora_rank=int(value.get("kv_lora_rank", 64)),
      qk_nope_head_dim=int(value.get("qk_nope_head_dim", 16)),
      qk_rope_head_dim=int(value.get("qk_rope_head_dim", 16)),
      v_head_dim=int(value.get("v_head_dim", 32)),
      rope_head_dim=_optional_int(value.get("rope_head_dim")),
      o_lora_rank=_optional_int(value.get("o_lora_rank")),
      o_groups=int(value.get("o_groups", 1)),
      attention_layer_types=_optional_str_tuple(value.get("attention_layer_types")),
      compress_rate_csa=int(value.get("compress_rate_csa", 4)),
      compress_rate_hca=int(value.get("compress_rate_hca", 128)),
      index_n_heads=int(value.get("index_n_heads", 64)),
      index_head_dim=int(value.get("index_head_dim", 128)),
      index_topk=int(value.get("index_topk", 512)),
      n_routed_experts=int(value.get("n_routed_experts", 4)),
      n_shared_experts=int(value.get("n_shared_experts", 1)),
      n_experts_per_token=int(value.get("n_experts_per_token", 2)),
      n_group=int(value.get("n_group", 2)),
      topk_group=int(value.get("topk_group", 1)),
      dense_layers=int(value.get("dense_layers", 1)),
    )
  raise ValueError(f"unsupported model.kind: {kind}")


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
  if value is None:
    return None
  if not isinstance(value, list | tuple):
    raise ValueError("attention_layer_types must be a list")
  return tuple(str(item) for item in value)


def _parse_sinks(value: Any) -> tuple[SinkConfig, ...]:
  if value is None:
    return ()
  if not isinstance(value, list | tuple):
    raise ValueError("sinks must be a list")
  return tuple(_parse_sink(item) for item in value)


def _parse_sink(value: Any) -> SinkConfig:
  if not isinstance(value, dict):
    raise ValueError("sink config must be a mapping")
  kind = value.get("kind")
  if kind == "files":
    return FilesSinkConfig(
      run_dir=Path(value["run_dir"]) if "run_dir" in value else None,
      config_json=str(value.get("config_json", "config.json")),
      resolved_config_yaml=str(
        value.get("resolved_config_yaml", "config.resolved.yaml")
      ),
      status_json=str(value.get("status_json", "status.json")),
      metrics_jsonl=str(value.get("metrics_jsonl", "metrics.jsonl")),
      result_json=str(value.get("result_json", "result.json")),
    )
  if kind == "tensorboard":
    return TensorBoardSinkConfig(
      log_dir=Path(value["log_dir"]) if "log_dir" in value else None,
      flush_secs=int(value.get("flush_secs", 10)),
    )
  if kind == "mlflow":
    return MlflowSinkConfig(
      tracking_uri=value.get("tracking_uri"),
      experiment_name=str(value.get("experiment_name", "flm")),
      run_name=value.get("run_name"),
      nested=bool(value.get("nested", False)),
    )
  if kind == "wandb":
    return WandbSinkConfig(
      project=str(value.get("project", "flm")),
      entity=value.get("entity"),
      name=value.get("name"),
      mode=value.get("mode"),
      dir=Path(value["dir"]) if "dir" in value else None,
      tags=_str_tuple(value.get("tags")),
      group=value.get("group"),
      job_type=value.get("job_type"),
    )
  raise ValueError(f"unsupported sink kind: {kind}")


def _str_tuple(value: Any) -> tuple[str, ...]:
  if value is None:
    return ()
  if not isinstance(value, list | tuple):
    raise ValueError("tags must be a list")
  return tuple(str(item) for item in value)
