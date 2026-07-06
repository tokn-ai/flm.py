"""Teaching scaffold for DeepSeek V4 compressed sparse attention."""

from __future__ import annotations

import torch
from torch import nn


class DeepSeekV4IndexerScorer(nn.Module):
  """Lightning-indexer scoring head scaffold."""

  def __init__(
    self,
    d_model: int,
    index_n_heads: int,
    index_head_dim: int,
  ) -> None:
    raise NotImplementedError

  def forward(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    hidden_states: torch.Tensor,
  ) -> torch.Tensor:
    raise NotImplementedError


class DeepSeekV4Indexer(nn.Module):
  """Stateless DeepSeek V4 Lightning Indexer scaffold."""

  def __init__(
    self,
    d_model: int,
    q_lora_rank: int,
    compress_rate: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope_base: float = 10_000.0,
    norm_eps: float = 1e-6,
  ) -> None:
    raise NotImplementedError

  def forward(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    positions: torch.Tensor,
  ) -> torch.Tensor:
    raise NotImplementedError


class DeepSeekV4CSACompressor(nn.Module):
  """Stateless DeepSeek V4 compressed sparse attention compressor scaffold."""

  def __init__(
    self,
    d_model: int,
    head_dim: int,
    q_lora_rank: int,
    compress_rate: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope_head_dim: int | None = None,
    rope_base: float = 10_000.0,
    norm_eps: float = 1e-6,
  ) -> None:
    raise NotImplementedError

  def forward(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    positions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError
