import pytest
import torch
from flm_datasets import TokenDataset


def test_token_dataset_returns_next_token_targets() -> None:
  dataset = TokenDataset([10, 11, 12, 13, 14], seq_len=3)

  x, y = dataset[1]

  torch.testing.assert_close(x, torch.tensor([11, 12, 13]))
  torch.testing.assert_close(y, torch.tensor([12, 13, 14]))


def test_token_dataset_rejects_short_token_streams() -> None:
  with pytest.raises(ValueError, match="greater than seq_len"):
    TokenDataset([1, 2], seq_len=2)
