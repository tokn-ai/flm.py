from collections.abc import Callable

import pytest
import torch


@pytest.fixture
def random_input() -> Callable[..., torch.Tensor]:
  torch.manual_seed(42)

  def make(*shape: int) -> torch.Tensor:
    return torch.randn(*shape)

  return make
