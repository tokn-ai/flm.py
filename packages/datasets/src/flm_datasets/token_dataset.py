"""Token datasets."""

from __future__ import annotations

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
