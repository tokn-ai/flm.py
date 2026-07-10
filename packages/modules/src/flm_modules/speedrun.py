"""Portable building blocks used by the nanoGPT speedrun model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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

  def forward(
    self,
    x: torch.Tensor,
    previous_embeddings: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if x.ndim != 3:
      raise ValueError("TokenSmear expects [batch, sequence, model] input")
    if previous_embeddings is not None:
      if previous_embeddings.shape != x.shape:
        raise ValueError("previous_embeddings must have the same shape as x")
      gate = torch.sigmoid(self.gate(x[..., : self.gate_dim]))
      return x + self.scale * gate * previous_embeddings
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

  def forward(
    self,
    input_ids: torch.Tensor,
    previous_input_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    hashes = self.hash_ids(input_ids, previous_input_ids)
    sign_ids = self.sign_ids(input_ids, previous_input_ids)
    return self.embedding(hashes) * self.sign_table[sign_ids]

  def hash_ids(
    self,
    input_ids: torch.Tensor,
    previous_input_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    self._validate_input(input_ids)
    modulus = self.num_embeddings - 1
    values = input_ids.to(torch.int64)
    if previous_input_ids is not None:
      previous = self._validate_previous(input_ids, previous_input_ids)
      hashes = torch.full_like(values, modulus)
      valid = previous >= 0
      hashes[valid] = torch.bitwise_xor(
        self.hash_multiplier_current * values[valid],
        self.hash_multiplier_previous * previous[valid],
      ).remainder(modulus)
      return hashes
    hashes = torch.full_like(values, modulus)
    hashes[:, 1:] = torch.bitwise_xor(
      self.hash_multiplier_current * values[:, 1:],
      self.hash_multiplier_previous * values[:, :-1],
    ).remainder(modulus)
    return hashes

  def sign_ids(
    self,
    input_ids: torch.Tensor,
    previous_input_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    self._validate_input(input_ids)
    values = input_ids.to(torch.int64)
    if previous_input_ids is not None:
      previous = self._validate_previous(input_ids, previous_input_ids)
      sign_ids = torch.zeros_like(values)
      valid = previous >= 0
      sign_ids[valid] = torch.bitwise_xor(
        previous[valid],
        values[valid],
      ).remainder(self.sign_table.shape[0])
      return sign_ids
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

  @staticmethod
  def _validate_previous(
    input_ids: torch.Tensor,
    previous_input_ids: torch.Tensor,
  ) -> torch.Tensor:
    if previous_input_ids.shape != input_ids.shape:
      raise ValueError("previous_input_ids must have the same shape as input_ids")
    return previous_input_ids.to(torch.int64)


class MultiwayDynamicDenseConnections(nn.Module):
  """Small per-token router for Multiway Dynamic Dense connections (MUDD)."""

  def __init__(
    self,
    d_model: int,
    *,
    hidden_dim: int = 64,
    num_routes: int = 2,
    max_coefficients: int = 14,
    output_scale: float = 0.1,
  ) -> None:
    super().__init__()
    if min(d_model, hidden_dim, num_routes, max_coefficients) < 1:
      raise ValueError("MUDD dimensions must be positive")
    if output_scale <= 0:
      raise ValueError("MUDD output_scale must be positive")
    self.max_coefficients = max_coefficients
    self.output_scale = output_scale
    self.up = nn.Parameter(torch.empty(num_routes, hidden_dim, d_model))
    self.down = nn.Parameter(torch.zeros(num_routes, max_coefficients, hidden_dim))
    self.bias = nn.Parameter(torch.zeros(num_routes, max_coefficients))
    nn.init.normal_(self.up, std=d_model**-0.5)

  def forward(
    self,
    x: torch.Tensor,
    *,
    route: int,
    num_coefficients: int,
  ) -> tuple[torch.Tensor, ...]:
    if not 0 <= route < self.up.shape[0]:
      raise ValueError("MUDD route is out of range")
    if not 1 <= num_coefficients <= self.max_coefficients:
      raise ValueError("invalid number of MUDD coefficients")
    hidden = F.gelu(F.linear(x, self.up[route]))
    coefficients = F.linear(
      hidden,
      self.down[route, :num_coefficients],
      self.bias[route, :num_coefficients],
    )
    return tuple((coefficients * self.output_scale).split(1, dim=-1))
