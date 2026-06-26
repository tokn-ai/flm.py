"""Teaching scaffold for DeepSeek sparse attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flm_modules.attentions.backends import AttentionBackend


class DeepSeekDSAIndexer(nn.Module):
  """DeepSeek V3.2 sparse attention indexer scaffold."""

  def __init__(
    self,
    d_model: int,
    q_lora_rank: int,
    qk_rope_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope_base: float = 10_000.0,
  ) -> None:
    super().__init__()
    raise NotImplementedError("Teaching DeepSeekDSAIndexer is not implemented")

  def forward(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    raise NotImplementedError("Teaching DeepSeekDSAIndexer is not implemented")

  def set_weights_from_transformers(
    self,
    wq_b: Tensor,
    wk: Tensor,
    k_norm_weight: Tensor,
    k_norm_bias: Tensor,
    weights_proj: Tensor,
  ) -> None:
    raise NotImplementedError("Teaching DeepSeekDSAIndexer is not implemented")


class DeepSeekDSA(nn.Module):
  """DeepSeek V3.2 sparse attention scaffold."""

  def __init__(
    self,
    d_model: int,
    n_heads: int,
    kv_lora_rank: int,
    q_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    bias: bool = False,
    rope_base: float = 10_000.0,
    norm_eps: float = 1e-6,
    causal: bool = True,
    backend: AttentionBackend | str = AttentionBackend.TORCH,
  ) -> None:
    super().__init__()
    raise NotImplementedError("Teaching DeepSeekDSA is not implemented")

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    raise NotImplementedError("Teaching DeepSeekDSA is not implemented")

  def set_weights_from_transformers(
    self,
    q_a_proj: Tensor,
    q_a_layernorm: Tensor,
    q_b_proj: Tensor,
    kv_a_proj_with_mqa: Tensor,
    kv_a_layernorm: Tensor,
    kv_b_proj: Tensor,
    o_proj: Tensor,
    indexer_wq_b: Tensor,
    indexer_wk: Tensor,
    indexer_k_norm_weight: Tensor,
    indexer_k_norm_bias: Tensor,
    indexer_weights_proj: Tensor,
  ) -> None:
    raise NotImplementedError("Teaching DeepSeekDSA is not implemented")
