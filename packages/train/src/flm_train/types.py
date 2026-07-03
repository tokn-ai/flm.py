"""Training configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LossBackend = Literal[
  "cross_entropy",
  "linear_cross_entropy",
  "cut_cross_entropy",
  "tilelang_linear_cross_entropy",
]
AttentionBackend = Literal["torch", "flash_attention2", "tilelang"]
TorchDType = Literal["float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class DataConfig:
  kind: Literal["token_dataset"] = "token_dataset"
  encoding_name: str = "cl100k_base"
  seq_len: int = 128
  dataset_root: Path = Path(".cache/data/repo_sources")
  version: str = "latest"
  split: Literal["train", "val", "test"] = "train"
  resolved_version: str | None = None


@dataclass(frozen=True)
class ReferenceModelConfig:
  kind: Literal["reference"] = "reference"
  d_model: int = 128
  n_layers: int = 2
  n_heads: int = 4
  d_ff: int | None = None
  attention_backend: AttentionBackend = "torch"
  loss_backend: LossBackend = "cross_entropy"
  loss_chunk_size: int = 512


@dataclass(frozen=True)
class DSTinyModelConfig:
  kind: Literal["ds_tiny"] = "ds_tiny"
  d_model: int = 128
  n_layers: int = 2
  n_heads: int = 4
  d_ff: int | None = None
  q_lora_rank: int | None = None
  kv_lora_rank: int = 64
  qk_nope_head_dim: int = 16
  qk_rope_head_dim: int = 16
  v_head_dim: int = 32
  attention_backend: AttentionBackend = "torch"
  loss_backend: LossBackend = "cross_entropy"
  loss_chunk_size: int = 512


@dataclass(frozen=True)
class DeepSeekV4ModelConfig:
  kind: Literal["deepseek_v4"] = "deepseek_v4"
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
  attention_backend: AttentionBackend = "torch"
  loss_backend: LossBackend = "cross_entropy"
  loss_chunk_size: int = 512


ModelConfig = ReferenceModelConfig | DSTinyModelConfig | DeepSeekV4ModelConfig


@dataclass(frozen=True)
class OptimizerConfig:
  kind: Literal["adamw"] = "adamw"
  learning_rate: float = 3e-4
  weight_decay: float = 0.1


@dataclass(frozen=True)
class LoopConfig:
  batch_size: int = 8
  steps: int = 10
  device: str = "cpu"
  seed: int = 42
  dtype: TorchDType = "float32"


@dataclass(frozen=True)
class EvalConfig:
  split: Literal["val", "test"] = "test"
  every_steps: int = 100
  max_batches: int = 8


@dataclass(frozen=True)
class RolloutPromptConfig:
  name: str
  prompt: str


@dataclass(frozen=True)
class RolloutConfig:
  every_steps: int = 100
  max_new_tokens: int = 64
  prompts: tuple[RolloutPromptConfig, ...] = ()


@dataclass(frozen=True)
class CheckpointConfig:
  enabled: bool = False
  every_steps: int = 100
  keep_last: int = 3
  resume: str | None = None


@dataclass(frozen=True)
class TrainConfig:
  data: DataConfig = field(default_factory=DataConfig)
  model: ModelConfig = field(default_factory=ReferenceModelConfig)
  optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
  loop: LoopConfig = field(default_factory=LoopConfig)
  eval: EvalConfig | None = None
  rollout: RolloutConfig | None = None
  checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass(frozen=True)
class TrainingResult:
  losses: list[float]
  token_count: int
  file_count: int
