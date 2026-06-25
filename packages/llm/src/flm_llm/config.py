"""Model configuration."""

from __future__ import annotations

from dataclasses import dataclass

from flm_modules import AttentionBackend


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

  @property
  def ffn_d_ff(self) -> int:
    if self.d_ff is not None:
      return self.d_ff
    return int(8 * self.d_model / 3)


@dataclass(frozen=True)
class DeepSeekV4Config:
  vocab_size: int
  max_seq_len: int = 2048
  d_model: int = 1024
  n_layers: int = 12
  n_heads: int = 16
  q_lora_rank: int | None = None
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 64
  qk_rope_head_dim: int = 64
  v_head_dim: int = 64
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

  @property
  def ffn_d_ff(self) -> int:
    if self.moe_d_ff is not None:
      return self.moe_d_ff
    return int(8 * self.d_model / 3)
