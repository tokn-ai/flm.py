"""Canonical binary FineWeb shards used by the nanoGPT speedrun."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FINEWEB_HEADER_INTS = 256
FINEWEB_HEADER_BYTES = FINEWEB_HEADER_INTS * 4
FINEWEB_MAGIC = 20240520
FINEWEB_VERSION = 1


def load_fineweb_binary(path: Path) -> np.memmap:
  """Validate and memory-map one canonical uint16 FineWeb token shard."""
  path = Path(path)
  if path.stat().st_size < FINEWEB_HEADER_BYTES:
    raise ValueError(f"FineWeb shard header is truncated: {path}")
  header = np.memmap(path, mode="r", dtype="<i4", shape=(FINEWEB_HEADER_INTS,))
  if int(header[0]) != FINEWEB_MAGIC:
    raise ValueError(f"FineWeb shard magic number mismatch: {path}")
  if int(header[1]) != FINEWEB_VERSION:
    raise ValueError(f"unsupported FineWeb shard version {int(header[1])}: {path}")
  token_count = int(header[2])
  if token_count < 0:
    raise ValueError(f"FineWeb shard token count must be non-negative: {path}")
  expected_size = FINEWEB_HEADER_BYTES + 2 * token_count
  if path.stat().st_size != expected_size:
    raise ValueError(
      f"FineWeb shard size mismatch: expected {expected_size} bytes, "
      f"found {path.stat().st_size}: {path}"
    )
  return np.memmap(
    path,
    mode="r",
    dtype="<u2",
    offset=FINEWEB_HEADER_BYTES,
    shape=(token_count,),
  )


class FineWebBinaryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
  """Non-overlapping next-token windows over ordered FineWeb binary shards.

  This matches speedrun validation semantics: window zero starts at token zero,
  no BOS alignment is performed, and each requested prediction token is scored
  exactly once.
  """

  def __init__(
    self,
    paths: Sequence[Path],
    *,
    seq_len: int,
    token_limit: int | None = None,
  ) -> None:
    if seq_len < 1:
      raise ValueError("seq_len must be positive")
    if not paths:
      raise ValueError("at least one FineWeb shard is required")
    self.shards = [load_fineweb_binary(path) for path in paths]
    self.cumulative_lengths: list[int] = []
    total = 0
    for shard in self.shards:
      total += len(shard)
      self.cumulative_lengths.append(total)
    available_predictions = total - 1
    if available_predictions < seq_len:
      raise ValueError("FineWeb token count must be greater than seq_len")
    if token_limit is None:
      token_limit = available_predictions - (available_predictions % seq_len)
    if token_limit < 1:
      raise ValueError("token_limit must be positive")
    if token_limit > available_predictions:
      raise ValueError("token_limit exceeds available next-token predictions")
    if token_limit % seq_len:
      raise ValueError("token_limit must be divisible by seq_len")
    self.seq_len = seq_len
    self.token_limit = token_limit

  def __len__(self) -> int:
    return self.token_limit // self.seq_len

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if index < 0 or index >= len(self):
      raise IndexError(index)
    start = index * self.seq_len
    tokens = self._slice(start, self.seq_len + 1)
    tensor = torch.from_numpy(tokens.astype(np.int64, copy=False))
    return tensor[:-1], tensor[1:]

  def _slice(self, start: int, length: int) -> np.ndarray:
    shard_index = bisect.bisect_right(self.cumulative_lengths, start)
    previous = 0 if shard_index == 0 else self.cumulative_lengths[shard_index - 1]
    local_start = start - previous
    remaining = length
    pieces = []
    while remaining:
      shard = self.shards[shard_index]
      take = min(remaining, len(shard) - local_start)
      pieces.append(shard[local_start : local_start + take])
      remaining -= take
      shard_index += 1
      local_start = 0
    if len(pieces) == 1:
      return pieces[0]
    return np.concatenate(pieces)
