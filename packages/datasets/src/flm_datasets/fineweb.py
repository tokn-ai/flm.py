"""Canonical binary FineWeb shards used by the nanoGPT speedrun."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

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
    self.source_token_count = sum(len(shard) for shard in self.shards)
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


class FineWebPackedDataset(
  IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
  """BOS-aligned document-fragment batches matching speedrun training order.

  Each yielded item is already a complete token-budget batch. Document fragments
  are separate padded rows, so eager causal attention cannot cross boundaries.
  Targets use ``-100`` only for padding and otherwise match the concatenated
  upstream stream, including the target BOS between adjacent fragments.
  """

  def __init__(
    self,
    paths: Sequence[Path],
    *,
    batch_tokens: int,
    max_seq_len: int,
    num_batches: int,
    bos_token_id: int = 50256,
  ) -> None:
    if not paths:
      raise ValueError("at least one FineWeb shard is required")
    if num_batches < 1:
      raise ValueError("num_batches must be positive")
    self.shards = [load_fineweb_binary(path) for path in paths]
    self.source_token_count = sum(len(shard) for shard in self.shards)
    self.num_batches = num_batches
    self.bos_token_id = bos_token_id
    self.set_batch_shape(batch_tokens=batch_tokens, max_seq_len=max_seq_len)

  def set_batch_shape(self, *, batch_tokens: int, max_seq_len: int) -> None:
    if batch_tokens < 1:
      raise ValueError("batch_tokens must be positive")
    if max_seq_len < 1:
      raise ValueError("max_seq_len must be positive")
    self.batch_tokens = batch_tokens
    self.max_seq_len = max_seq_len

  def __len__(self) -> int:
    return self.num_batches

  def __iter__(self):
    shard_index = 0
    bos_index = 0
    bos_positions = self._bos_positions(shard_index)
    yielded = 0
    while yielded < self.num_batches:
      packed = self._pack_from_shard(
        self.shards[shard_index],
        bos_positions,
        bos_index,
      )
      if packed is None:
        shard_index += 1
        if shard_index >= len(self.shards):
          return
        bos_index = 0
        bos_positions = self._bos_positions(shard_index)
        continue
      input_ids, targets, previous_input_ids, bos_index = packed
      yielded += 1
      yield input_ids, targets, previous_input_ids

  def _bos_positions(self, shard_index: int) -> np.ndarray:
    return np.flatnonzero(self.shards[shard_index] == self.bos_token_id).astype(
      np.int64,
      copy=False,
    )

  def _pack_from_shard(
    self,
    tokens: np.ndarray,
    bos_positions: np.ndarray,
    bos_index: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    starts = []
    ends = []
    total = 0
    index = bos_index
    while total <= self.batch_tokens:
      if index >= len(bos_positions):
        return None
      start = int(bos_positions[index])
      starts.append(start)
      index += 1
      next_bos = (
        int(bos_positions[index]) if index < len(bos_positions) else len(tokens)
      )
      end = min(
        next_bos,
        start + self.max_seq_len,
        start + self.batch_tokens - total + 1,
      )
      ends.append(end)
      total += end - start

    buffer = np.concatenate(
      [tokens[start:end] for start, end in zip(starts, ends, strict=True)]
    )
    flat_inputs = buffer[:-1].astype(np.int64, copy=False)
    flat_targets = buffer[1:].astype(np.int64, copy=False)
    flat_previous = np.empty_like(flat_inputs)
    flat_previous[0] = -1
    flat_previous[1:] = flat_inputs[:-1]
    lengths = [end - start for start, end in zip(starts, ends, strict=True)]
    lengths[-1] -= 1
    row_count = len(lengths)
    width = max(lengths)
    input_ids = torch.full((row_count, width), self.bos_token_id, dtype=torch.long)
    targets = torch.full((row_count, width), -100, dtype=torch.long)
    previous_input_ids = torch.full((row_count, width), -1, dtype=torch.long)
    offset = 0
    for row, length in enumerate(lengths):
      input_ids[row, :length] = torch.from_numpy(flat_inputs[offset : offset + length])
      targets[row, :length] = torch.from_numpy(flat_targets[offset : offset + length])
      previous_input_ids[row, :length] = torch.from_numpy(
        flat_previous[offset : offset + length]
      )
      offset += length
    if offset != self.batch_tokens:
      raise RuntimeError("packed FineWeb batch did not consume its token budget")
    return input_ids, targets, previous_input_ids, index


class FineWebValidationDataset(
  IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
  """Unaligned fixed-token validation chunks isolated at canonical BOS tokens."""

  def __init__(
    self,
    paths: Sequence[Path],
    *,
    batch_tokens: int,
    token_limit: int | None,
    bos_token_id: int = 50256,
  ) -> None:
    self.source = FineWebBinaryDataset(
      paths,
      seq_len=batch_tokens,
      token_limit=token_limit,
    )
    self.token_limit = self.source.token_limit
    self.batch_tokens = batch_tokens
    self.bos_token_id = bos_token_id

  def __len__(self) -> int:
    return len(self.source)

  def __iter__(self):
    for index in range(len(self.source)):
      flat_inputs, flat_targets = self.source[index]
      yield self._split_documents(flat_inputs, flat_targets)

  def _split_documents(
    self,
    flat_inputs: torch.Tensor,
    flat_targets: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bos_positions = torch.nonzero(flat_inputs == self.bos_token_id).flatten()
    boundaries = [0]
    boundaries.extend(int(position) for position in bos_positions if position > 0)
    boundaries.append(flat_inputs.numel())
    lengths = [
      end - start for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    width = max(lengths)
    input_ids = torch.full(
      (len(lengths), width),
      self.bos_token_id,
      dtype=torch.long,
    )
    targets = torch.full_like(input_ids, -100)
    previous_input_ids = torch.full_like(input_ids, -1)
    flat_previous = torch.empty_like(flat_inputs)
    flat_previous[0] = -1
    flat_previous[1:] = flat_inputs[:-1]
    for row, (start, end) in enumerate(
      zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
      length = end - start
      input_ids[row, :length] = flat_inputs[start:end]
      targets[row, :length] = flat_targets[start:end]
      previous_input_ids[row, :length] = flat_previous[start:end]
    return input_ids, targets, previous_input_ids
