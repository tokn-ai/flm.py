from pathlib import Path

import numpy as np
import pytest
import torch
from flm_datasets import (
  FINEWEB_HEADER_INTS,
  FINEWEB_MAGIC,
  FINEWEB_VERSION,
  FineWebBinaryDataset,
  load_fineweb_binary,
)


def _write_shard(path: Path, tokens: list[int], *, magic: int = FINEWEB_MAGIC) -> None:
  header = np.zeros(FINEWEB_HEADER_INTS, dtype="<i4")
  header[0] = magic
  header[1] = FINEWEB_VERSION
  header[2] = len(tokens)
  path.write_bytes(header.tobytes() + np.asarray(tokens, dtype="<u2").tobytes())


def test_load_fineweb_binary_validates_and_maps_tokens(tmp_path: Path) -> None:
  path = tmp_path / "fineweb_val_000000.bin"
  _write_shard(path, [50256, 10, 11, 12])

  tokens = load_fineweb_binary(path)

  assert isinstance(tokens, np.memmap)
  assert tokens.tolist() == [50256, 10, 11, 12]


def test_load_fineweb_binary_rejects_invalid_magic(tmp_path: Path) -> None:
  path = tmp_path / "bad.bin"
  _write_shard(path, [1, 2], magic=7)

  with pytest.raises(ValueError, match="magic number mismatch"):
    load_fineweb_binary(path)


def test_fineweb_binary_dataset_scores_ordered_non_overlapping_tokens(
  tmp_path: Path,
) -> None:
  first = tmp_path / "fineweb_val_000000.bin"
  second = tmp_path / "fineweb_val_000001.bin"
  _write_shard(first, [10, 11, 12, 13, 14])
  _write_shard(second, [15, 16, 17, 18])
  dataset = FineWebBinaryDataset([first, second], seq_len=4, token_limit=8)

  x0, y0 = dataset[0]
  x1, y1 = dataset[1]

  torch.testing.assert_close(x0, torch.tensor([10, 11, 12, 13]))
  torch.testing.assert_close(y0, torch.tensor([11, 12, 13, 14]))
  torch.testing.assert_close(x1, torch.tensor([14, 15, 16, 17]))
  torch.testing.assert_close(y1, torch.tensor([15, 16, 17, 18]))


def test_fineweb_binary_dataset_requires_exact_window_limit(tmp_path: Path) -> None:
  path = tmp_path / "fineweb_val_000000.bin"
  _write_shard(path, list(range(10)))

  with pytest.raises(ValueError, match="divisible by seq_len"):
    FineWebBinaryDataset([path], seq_len=4, token_limit=7)
