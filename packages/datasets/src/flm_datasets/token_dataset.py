"""Token datasets."""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
  def __init__(self, tokens: Sequence[int] | np.ndarray, seq_len: int) -> None:
    if seq_len < 1:
      raise ValueError("seq_len must be positive")
    if len(tokens) <= seq_len:
      raise ValueError("token count must be greater than seq_len")

    if isinstance(tokens, np.ndarray):
      self.tokens = torch.from_numpy(tokens)
    else:
      self.tokens = torch.tensor(tokens, dtype=torch.long)
    self.seq_len = seq_len

  def __len__(self) -> int:
    return self.tokens.numel() - self.seq_len

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if index < 0 or index >= len(self):
      raise IndexError(index)
    x = self.tokens[index : index + self.seq_len].long()
    y = self.tokens[index + 1 : index + self.seq_len + 1].long()
    return x, y


class ShardedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
  def __init__(self, shards: Sequence[np.ndarray], seq_len: int) -> None:
    if seq_len < 1:
      raise ValueError("seq_len must be positive")
    self.shards = [torch.from_numpy(shard) for shard in shards if len(shard) > seq_len]
    self.seq_len = seq_len
    lengths = [shard.numel() - seq_len for shard in self.shards]
    if sum(lengths) <= 0:
      raise ValueError("token count must be greater than seq_len")
    self.cumulative_lengths: list[int] = []
    total = 0
    for length in lengths:
      total += int(length)
      self.cumulative_lengths.append(total)

  def __len__(self) -> int:
    return self.cumulative_lengths[-1]

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if index < 0 or index >= len(self):
      raise IndexError(index)
    shard_index = bisect.bisect_right(self.cumulative_lengths, index)
    previous = 0 if shard_index == 0 else self.cumulative_lengths[shard_index - 1]
    local_index = index - previous
    shard = self.shards[shard_index]
    x = shard[local_index : local_index + self.seq_len].long()
    y = shard[local_index + 1 : local_index + self.seq_len + 1].long()
    return x, y
