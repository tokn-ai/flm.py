"""Experiment configuration types and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from flm_train.train import TrainConfig


@dataclass(frozen=True)
class DataConfig:
  kind: Literal["repo_sources"] = "repo_sources"
  repo_root: Path = Path(".")
  encoding_name: str = "cl100k_base"
  seq_len: int = 128


@dataclass(frozen=True)
class ModelConfig:
  name: Literal["reference", "deepseek_v4", "ds_tiny"] = "reference"
  d_model: int = 128
  n_layers: int = 2
  n_heads: int = 4
  head_dim: int | None = None
  d_ff: int | None = None
  q_lora_rank: int | None = None
  kv_lora_rank: int = 64
  qk_nope_head_dim: int = 16
  qk_rope_head_dim: int = 16
  v_head_dim: int = 32
  rope_head_dim: int | None = None
  o_lora_rank: int | None = None
  o_groups: int = 1
  attention_layer_types: tuple[str, ...] | None = None
  compress_rate_csa: int = 4
  compress_rate_hca: int = 128
  index_n_heads: int = 64
  index_head_dim: int = 128
  index_topk: int = 512
  n_routed_experts: int = 4
  n_shared_experts: int = 1
  n_experts_per_token: int = 2
  n_group: int = 2
  topk_group: int = 1
  dense_layers: int = 1


@dataclass(frozen=True)
class OptimizerConfig:
  name: Literal["adamw"] = "adamw"
  learning_rate: float = 3e-4
  weight_decay: float = 0.1


@dataclass(frozen=True)
class RunTrainConfig:
  batch_size: int = 8
  steps: int = 10


@dataclass(frozen=True)
class OutputConfig:
  run_dir: Path | None = None


@dataclass(frozen=True)
class ExperimentConfig:
  name: str
  seed: int = 42
  device: str = "cpu"
  data: DataConfig = field(default_factory=DataConfig)
  model: ModelConfig = field(default_factory=ModelConfig)
  optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
  train: RunTrainConfig = field(default_factory=RunTrainConfig)
  output: OutputConfig = field(default_factory=OutputConfig)

  @property
  def run_dir(self) -> Path:
    if self.output.run_dir is not None:
      return self.output.run_dir
    return Path("runs") / self.name

  def to_train_config(self) -> TrainConfig:
    if self.data.kind != "repo_sources":
      raise ValueError(f"unsupported data.kind: {self.data.kind}")
    if self.optimizer.name != "adamw":
      raise ValueError(f"unsupported optimizer.name: {self.optimizer.name}")
    return TrainConfig(
      repo_root=self.data.repo_root,
      model_name=self.model.name,
      encoding_name=self.data.encoding_name,
      seq_len=self.data.seq_len,
      batch_size=self.train.batch_size,
      steps=self.train.steps,
      learning_rate=self.optimizer.learning_rate,
      weight_decay=self.optimizer.weight_decay,
      d_model=self.model.d_model,
      n_layers=self.model.n_layers,
      n_heads=self.model.n_heads,
      head_dim=self.model.head_dim,
      d_ff=self.model.d_ff,
      q_lora_rank=self.model.q_lora_rank,
      kv_lora_rank=self.model.kv_lora_rank,
      qk_nope_head_dim=self.model.qk_nope_head_dim,
      qk_rope_head_dim=self.model.qk_rope_head_dim,
      v_head_dim=self.model.v_head_dim,
      rope_head_dim=self.model.rope_head_dim,
      o_lora_rank=self.model.o_lora_rank,
      o_groups=self.model.o_groups,
      attention_layer_types=self.model.attention_layer_types,
      compress_rate_csa=self.model.compress_rate_csa,
      compress_rate_hca=self.model.compress_rate_hca,
      index_n_heads=self.model.index_n_heads,
      index_head_dim=self.model.index_head_dim,
      index_topk=self.model.index_topk,
      n_routed_experts=self.model.n_routed_experts,
      n_shared_experts=self.model.n_shared_experts,
      n_experts_per_token=self.model.n_experts_per_token,
      n_group=self.model.n_group,
      topk_group=self.model.topk_group,
      dense_layers=self.model.dense_layers,
      device=self.device,
      seed=self.seed,
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
  allowed = {"name", "seed", "device", "data", "model", "optimizer", "train", "output"}
  unknown = set(raw) - allowed
  if unknown:
    raise ValueError(f"unknown experiment config keys: {sorted(unknown)}")
  if "name" not in raw:
    raise ValueError("experiment config requires 'name'")

  data = _section(raw, "data")
  model = _section(raw, "model")
  optimizer = _section(raw, "optimizer")
  train = _section(raw, "train")
  output = _section(raw, "output")

  return ExperimentConfig(
    name=str(raw["name"]),
    seed=int(raw.get("seed", 42)),
    device=str(raw.get("device", "cpu")),
    data=DataConfig(
      kind=data.get("kind", "repo_sources"),
      repo_root=Path(data.get("repo_root", ".")),
      encoding_name=str(data.get("encoding_name", "cl100k_base")),
      seq_len=int(data.get("seq_len", 128)),
    ),
    model=ModelConfig(
      name=model.get("name", "reference"),
      d_model=int(model.get("d_model", 128)),
      n_layers=int(model.get("n_layers", 2)),
      n_heads=int(model.get("n_heads", 4)),
      head_dim=_optional_int(model.get("head_dim")),
      d_ff=_optional_int(model.get("d_ff")),
      q_lora_rank=_optional_int(model.get("q_lora_rank")),
      kv_lora_rank=int(model.get("kv_lora_rank", 64)),
      qk_nope_head_dim=int(model.get("qk_nope_head_dim", 16)),
      qk_rope_head_dim=int(model.get("qk_rope_head_dim", 16)),
      v_head_dim=int(model.get("v_head_dim", 32)),
      rope_head_dim=_optional_int(model.get("rope_head_dim")),
      o_lora_rank=_optional_int(model.get("o_lora_rank")),
      o_groups=int(model.get("o_groups", 1)),
      attention_layer_types=_optional_str_tuple(model.get("attention_layer_types")),
      compress_rate_csa=int(model.get("compress_rate_csa", 4)),
      compress_rate_hca=int(model.get("compress_rate_hca", 128)),
      index_n_heads=int(model.get("index_n_heads", 64)),
      index_head_dim=int(model.get("index_head_dim", 128)),
      index_topk=int(model.get("index_topk", 512)),
      n_routed_experts=int(model.get("n_routed_experts", 4)),
      n_shared_experts=int(model.get("n_shared_experts", 1)),
      n_experts_per_token=int(model.get("n_experts_per_token", 2)),
      n_group=int(model.get("n_group", 2)),
      topk_group=int(model.get("topk_group", 1)),
      dense_layers=int(model.get("dense_layers", 1)),
    ),
    optimizer=OptimizerConfig(
      name=optimizer.get("name", "adamw"),
      learning_rate=float(optimizer.get("learning_rate", 3e-4)),
      weight_decay=float(optimizer.get("weight_decay", 0.1)),
    ),
    train=RunTrainConfig(
      batch_size=int(train.get("batch_size", 8)),
      steps=int(train.get("steps", 10)),
    ),
    output=OutputConfig(
      run_dir=Path(output["run_dir"]) if "run_dir" in output else None,
    ),
  )


def apply_overrides(
  config: ExperimentConfig,
  overrides: ExperimentOverrides,
) -> ExperimentConfig:
  return ExperimentConfig(
    name=config.name,
    seed=config.seed if overrides.seed is None else overrides.seed,
    device=config.device if overrides.device is None else overrides.device,
    data=config.data,
    model=config.model,
    optimizer=config.optimizer,
    train=config.train
    if overrides.steps is None
    else RunTrainConfig(batch_size=config.train.batch_size, steps=overrides.steps),
    output=config.output
    if overrides.run_dir is None
    else OutputConfig(run_dir=overrides.run_dir),
  )


def config_to_plain(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, tuple):
    return list(value)
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


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
  if value is None:
    return None
  if not isinstance(value, list | tuple):
    raise ValueError("attention_layer_types must be a list")
  return tuple(str(item) for item in value)
