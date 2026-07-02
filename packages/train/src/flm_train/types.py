"""Training configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DataTrainConfig:
  kind: Literal["repo_sources"] = "repo_sources"
  repo_root: Path = Path(".")
  encoding_name: str = "cl100k_base"
  seq_len: int = 128


@dataclass(frozen=True)
class ModelTrainConfig:
  kind: Literal["reference", "deepseek_v4", "ds_tiny"] = "reference"
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
class OptimizerTrainConfig:
  kind: Literal["adamw"] = "adamw"
  learning_rate: float = 3e-4
  weight_decay: float = 0.1


@dataclass(frozen=True)
class LoopTrainConfig:
  batch_size: int = 8
  steps: int = 10
  device: str = "cpu"
  seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
  data: DataTrainConfig = field(default_factory=DataTrainConfig)
  model: ModelTrainConfig = field(default_factory=ModelTrainConfig)
  optimizer: OptimizerTrainConfig = field(default_factory=OptimizerTrainConfig)
  loop: LoopTrainConfig = field(default_factory=LoopTrainConfig)


@dataclass(frozen=True)
class TrainingResult:
  losses: list[float]
  token_count: int
  file_count: int
