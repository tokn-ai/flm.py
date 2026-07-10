"""Model configuration."""

from __future__ import annotations

from dataclasses import dataclass

from flm_modules import AttentionBackend, DeepSeekV4AttentionKind
from flm_modules.losses import LossBackend


@dataclass(frozen=True)
class ReferenceModelConfig:
  vocab_size: int
  max_seq_len: int = 2048
  d_model: int = 768
  n_layers: int = 12
  n_heads: int = 12
  d_ff: int | None = None
  bias: bool = False
  rope_base: float = 10_000.0
  norm_eps: float = 1e-6
  attention_backend: AttentionBackend | str = AttentionBackend.TORCH
  loss_backend: LossBackend = "cross_entropy"
  loss_chunk_size: int = 512

  @property
  def ffn_d_ff(self) -> int:
    if self.d_ff is not None:
      return self.d_ff
    return int(8 * self.d_model / 3)


@dataclass(frozen=True)
class NanoGPTSpeedrunConfig:
  """Eager reference configuration for the nanoGPT short-track model."""

  vocab_size: int = 50_304
  max_seq_len: int = 1024
  d_model: int = 768
  n_layers: int = 11
  n_heads: int = 12
  d_ff: int = 3072
  bias: bool = False
  rope_base: float = 10_000.0
  norm_eps: float = 1e-6
  attention_backend: AttentionBackend | str = AttentionBackend.TORCH
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
  bigram_vocab_size: int | None = None
  bigram_dim: int = 192
  bigram_sign_table_rows: int = 8192
  embedding_skip: bool = True
  value_residual: bool = True
  block_skip_from: int | None = 3
  block_skip_to: int | None = 6
  residual_decay: float = 1.0
  tie_embeddings: bool = True


@dataclass(frozen=True)
class DSTinyConfig:
  vocab_size: int
  max_seq_len: int = 2048
  d_model: int = 768
  n_layers: int = 12
  n_heads: int = 12
  q_lora_rank: int | None = None
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128
  d_ff: int | None = None
  bias: bool = False
  rope_base: float = 10_000.0
  norm_eps: float = 1e-6
  attention_backend: AttentionBackend | str = AttentionBackend.TORCH
  loss_backend: LossBackend = "cross_entropy"
  loss_chunk_size: int = 512

  @property
  def ffn_d_ff(self) -> int:
    if self.d_ff is not None:
      return self.d_ff
    return int(8 * self.d_model / 3)

  @property
  def attention_q_lora_rank(self) -> int | None:
    return self.q_lora_rank


@dataclass(frozen=True)
class DeepSeekV4Config:
  vocab_size: int
  max_seq_len: int = 2048
  d_model: int = 1024
  n_layers: int = 12
  n_heads: int = 16
  head_dim: int | None = None
  q_lora_rank: int | None = None
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 64
  qk_rope_head_dim: int = 64
  v_head_dim: int = 64
  rope_head_dim: int | None = None
  o_lora_rank: int | None = None
  o_groups: int = 1
  attention_layer_types: tuple[DeepSeekV4AttentionKind | str, ...] | None = None
  compress_rate_csa: int = 4
  compress_rate_hca: int = 128
  index_n_heads: int = 64
  index_head_dim: int = 128
  index_topk: int = 512
  moe_d_ff: int | None = None
  n_routed_experts: int = 8
  n_shared_experts: int = 1
  n_experts_per_token: int = 2
  n_group: int = 4
  topk_group: int = 1
  norm_topk_prob: bool = True
  routed_scaling_factor: float = 1.0
  dense_layers: int = 1
  hc_mult: int = 2
  hc_sinkhorn_iters: int = 3
  hc_eps: float = 1e-6
  bias: bool = False
  rope_base: float = 10_000.0
  norm_eps: float = 1e-6
  initializer_range: float = 0.02
  attention_backend: AttentionBackend | str = AttentionBackend.TORCH
  loss_backend: LossBackend = "cross_entropy"
  loss_chunk_size: int = 512

  @property
  def ffn_d_ff(self) -> int:
    if self.moe_d_ff is not None:
      return self.moe_d_ff
    return int(8 * self.d_model / 3)

  @property
  def attention_head_dim(self) -> int:
    if self.head_dim is not None:
      return self.head_dim
    return self.v_head_dim

  @property
  def attention_rope_head_dim(self) -> int:
    if self.rope_head_dim is not None:
      return self.rope_head_dim
    return self.attention_head_dim

  @property
  def attention_q_lora_rank(self) -> int:
    if self.q_lora_rank is not None:
      return self.q_lora_rank
    return max(1, self.d_model // 2)

  @property
  def attention_o_lora_rank(self) -> int:
    if self.o_lora_rank is not None:
      return self.o_lora_rank
    return max(1, self.d_model // 4)

  def attention_layer_type(self, layer_idx: int) -> DeepSeekV4AttentionKind:
    if not self.attention_layer_types:
      return DeepSeekV4AttentionKind.SLIDING
    return DeepSeekV4AttentionKind(
      self.attention_layer_types[layer_idx % len(self.attention_layer_types)]
    )
