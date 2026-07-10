from pathlib import Path

import numpy as np
import pytest
import torch
from flm_datasets import (
  FINEWEB_HEADER_INTS,
  FINEWEB_MAGIC,
  FINEWEB_VERSION,
  FineWebBinaryDataset,
  FineWebPackedDataset,
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


def test_fineweb_packed_dataset_aligns_and_truncates_documents(tmp_path: Path) -> None:
  path = tmp_path / "fineweb_train_000001.bin"
  _write_shard(path, [99, 1, 2, 3, 4, 99, 5, 6, 99, 7, 8, 9, 99, 10, 11])
  dataset = FineWebPackedDataset(
    [path],
    batch_tokens=6,
    max_seq_len=4,
    num_batches=2,
    bos_token_id=99,
  )

  batches = list(dataset)

  input_ids, targets, previous = batches[0]
  torch.testing.assert_close(
    input_ids,
    torch.tensor([[99, 1, 2, 3], [99, 5, 99, 99]]),
  )
  torch.testing.assert_close(
    targets,
    torch.tensor([[1, 2, 3, 99], [5, 6, -100, -100]]),
  )
  torch.testing.assert_close(
    previous,
    torch.tensor([[-1, 99, 1, 2], [3, 99, -1, -1]]),
  )
  next_inputs, next_targets, _ = batches[1]
  torch.testing.assert_close(
    next_inputs,
    torch.tensor([[99, 7, 8, 9], [99, 10, 99, 99]]),
  )
  torch.testing.assert_close(
    next_targets,
    torch.tensor([[7, 8, 9, 99], [10, 11, -100, -100]]),
  )


def test_fineweb_packed_dataset_accepts_runtime_batch_shape_changes(
  tmp_path: Path,
) -> None:
  path = tmp_path / "fineweb_train_000001.bin"
  _write_shard(
    path,
    [99, 1, 2, 99, 3, 4, 99, 5, 6, 99, 7, 8, 99, 9, 10, 99, 11, 12],
  )
  dataset = FineWebPackedDataset(
    [path],
    batch_tokens=3,
    max_seq_len=3,
    num_batches=2,
    bos_token_id=99,
  )
  iterator = iter(dataset)

  first_inputs, _, _ = next(iterator)
  dataset.set_batch_shape(batch_tokens=5, max_seq_len=2)
  second_inputs, second_targets, _ = next(iterator)

  assert first_inputs.shape == (2, 3)
  assert second_inputs.shape == (3, 2)
  assert int((second_targets != -100).sum()) == 5
