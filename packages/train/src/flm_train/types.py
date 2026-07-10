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
BatchSize = int | Literal["auto"]


@dataclass(frozen=True)
class DataConfig:
  kind: Literal["token_dataset"] = "token_dataset"
  encoding_name: str = "cl100k_base"
  seq_len: int = 128
  dataset_root: Path = Path("cache/repo_sources_cl100k")
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
class NanoGPTSpeedrunModelConfig:
  kind: Literal["nanogpt_speedrun"] = "nanogpt_speedrun"
  d_model: int = 768
  n_layers: int = 11
  n_heads: int = 12
  d_ff: int = 3072
  attention_backend: AttentionBackend = "torch"
  loss_backend: LossBackend = "cross_entropy"
  loss_chunk_size: int = 512
  logit_softcap: float | None = 30.0
  logit_scale: float = 1.0
  logit_sigmoid_scale: float | None = 23.0
  logit_sigmoid_bias: float = 5.0
  logit_sigmoid_temperature: float = 7.5
  token_smear: bool = True
  smear_gate_dim: int = 12
  partial_key_offset_layers: tuple[int, ...] = (3, 10)
  attention_gate_dim: int = 12
  xsa: bool = True
  attention_free_layer: int | None = 6
  paired_head_layers: tuple[int, ...] = (0, 2, 5, 9)
  long_window_layers: tuple[int, ...] = (3, 10)
  shared_attention_source_layer: int | None = 7
  shared_attention_start_layer: int | None = 8
  value_embedding_layers: tuple[int, ...] = (1, 2, 8, 9, 10)
  value_embedding_gate_dim: int = 12
  mudd: bool = True
  mudd_hidden_dim: int = 64
  mudd_scale: float = 0.1
  bigram_vocab_size: int | None = None
  bigram_dim: int = 192
  bigram_sign_table_rows: int = 8192
  mtp_weights: tuple[float, ...] = (1.0, 0.5, 0.25)
  embedding_skip: bool = True
  value_residual: bool = True
  block_skip_from: int | None = 3
  block_skip_to: int | None = 6
  residual_decay: float = 1.1
  tie_embeddings: bool = True


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


ModelConfig = (
  ReferenceModelConfig
  | NanoGPTSpeedrunModelConfig
  | DSTinyModelConfig
  | DeepSeekV4ModelConfig
)


@dataclass(frozen=True)
class OptimizerConfig:
  kind: Literal["adamw", "muon", "normuon", "speedrun_normuon"] = "adamw"
  learning_rate: float = 3e-4
  weight_decay: float = 0.1
  max_grad_norm: float | None = 1.0
  secondary_update_every: int = 1


@dataclass(frozen=True)
class OptimizerScheduleConfig:
  warmup_steps: int = 0
  cooldown_steps: int = 0
  final_lr_scale: float = 0.0
  momentum_start: float | None = None
  momentum_end: float | None = None
  momentum_warmup_steps: int = 0
  scale_weight_decay_with_lr: bool = False


@dataclass(frozen=True)
class SpeedrunStageConfig:
  end_step: int
  batch_size: int | None = None
  seq_len: int | None = None
  learning_rate_scale: float = 1.0
  mtp_weights: tuple[float, ...] | None = None
  short_window: int | None = None
  long_window: int | None = None


@dataclass(frozen=True)
class SpeedrunScheduleConfig:
  stages: tuple[SpeedrunStageConfig, ...] = ()
  untie_step: int | None = None


@dataclass(frozen=True)
class LoopConfig:
  batch_size: BatchSize = 8
  batch_size_vram_fraction: float = 0.9
  steps: int = 10
  device: str = "cpu"
  seed: int = 42
  dtype: TorchDType = "float32"
  gradient_accumulation_steps: int = 1


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
  schedule: OptimizerScheduleConfig = field(default_factory=OptimizerScheduleConfig)
  speedrun_schedule: SpeedrunScheduleConfig = field(
    default_factory=SpeedrunScheduleConfig
  )
  loop: LoopConfig = field(default_factory=LoopConfig)
  eval: EvalConfig | None = None
  rollout: RolloutConfig | None = None
  checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass(frozen=True)
class TrainingResult:
  losses: list[float]
  token_count: int
  file_count: int
  byte_count: int
