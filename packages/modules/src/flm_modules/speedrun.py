"""Portable building blocks used by the nanoGPT speedrun model."""

from __future__ import annotations

import torch
from torch import nn


class TokenSmear(nn.Module):
  """Inject a gated copy of the previous token into the residual stream."""

  def __init__(self, d_model: int, gate_dim: int = 12) -> None:
    super().__init__()
    if not 1 <= gate_dim <= d_model:
      raise ValueError("gate_dim must be in [1, d_model]")
    self.gate_dim = gate_dim
    self.gate = nn.Linear(gate_dim, 1, bias=False)
    self.scale = nn.Parameter(torch.zeros(()))
    nn.init.zeros_(self.gate.weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
      raise ValueError("TokenSmear expects [batch, sequence, model] input")
    if x.shape[1] < 2:
      return x
    gate = torch.sigmoid(self.gate(x[:, 1:, : self.gate_dim]))
    smeared = x[:, 1:] + self.scale * gate * x[:, :-1]
    return torch.cat((x[:, :1], smeared), dim=1)


class BigramHashEmbedding(nn.Module):
  """Hashed previous/current-token embedding with a collision sign trick."""

  def __init__(
    self,
    num_embeddings: int,
    embedding_dim: int,
    *,
    sign_table_rows: int = 8192,
    hash_multiplier_current: int = 36_313,
    hash_multiplier_previous: int = 27_191,
  ) -> None:
    super().__init__()
    if num_embeddings < 2:
      raise ValueError("num_embeddings must be at least 2")
    if sign_table_rows < 1:
      raise ValueError("sign_table_rows must be positive")
    self.num_embeddings = num_embeddings
    self.hash_multiplier_current = hash_multiplier_current
    self.hash_multiplier_previous = hash_multiplier_previous
    self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.zeros_(self.embedding.weight)
    self.register_buffer(
      "sign_table",
      torch.randn(sign_table_rows, embedding_dim).sign(),
    )

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    hashes = self.hash_ids(input_ids)
    sign_ids = self.sign_ids(input_ids)
    return self.embedding(hashes) * self.sign_table[sign_ids]

  def hash_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    self._validate_input(input_ids)
    modulus = self.num_embeddings - 1
    values = input_ids.to(torch.int64)
    hashes = torch.full_like(values, modulus)
    hashes[:, 1:] = torch.bitwise_xor(
      self.hash_multiplier_current * values[:, 1:],
      self.hash_multiplier_previous * values[:, :-1],
    ).remainder(modulus)
    return hashes

  def sign_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    self._validate_input(input_ids)
    values = input_ids.to(torch.int64)
    sign_ids = torch.zeros_like(values)
    sign_ids[:, 1:] = torch.bitwise_xor(
      values[:, :-1],
      values[:, 1:],
    ).remainder(self.sign_table.shape[0])
    return sign_ids

  @staticmethod
  def _validate_input(input_ids: torch.Tensor) -> None:
    if input_ids.ndim != 2:
      raise ValueError("input_ids must have shape [batch, sequence]")
