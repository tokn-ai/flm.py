"""Model configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceModelConfig:
  vocab_size: int
  max_seq_len: int = 2048
  d_model: int = 768
  n_layers: int = 12
  n_heads: int = 12
  d_ff: int | None = None
  dropout: float = 0.0
  bias: bool = False
  rope_base: float = 10_000.0
  norm_eps: float = 1e-6

  @property
  def ffn_d_ff(self) -> int:
    if self.d_ff is not None:
      return self.d_ff
    return int(8 * self.d_model / 3)
