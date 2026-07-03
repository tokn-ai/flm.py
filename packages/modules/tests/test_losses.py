from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from flm_modules.losses import language_model_loss, linear_cross_entropy
from torch.nn import functional as F


@dataclass(frozen=True)
class LossCase:
  name: str
  loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def test_linear_cross_entropy_matches_cross_entropy() -> None:
  _assert_matches_cross_entropy(
    LossCase(
      name="chunked_cce",
      loss_fn=lambda hidden, weight, targets: linear_cross_entropy(
        hidden_states=hidden,
        classifier_weight=weight,
        targets=targets,
        chunk_size=4,
      ),
    )
  )


def test_torch_linear_cross_entropy_matches_cross_entropy_when_available() -> None:
  torch_linear_cross_entropy = getattr(F, "linear_cross_entropy", None)
  if torch_linear_cross_entropy is None:
    pytest.skip("torch.nn.functional.linear_cross_entropy is not available")
  _assert_matches_cross_entropy(
    LossCase(
      name="F.linear_cross_entropy",
      loss_fn=lambda hidden, weight, targets: torch_linear_cross_entropy(
        hidden,
        weight,
        targets,
      ),
    )
  )


def test_cut_cross_entropy_matches_cross_entropy_when_available() -> None:
  cut_cross_entropy = _cut_cross_entropy()
  if cut_cross_entropy is None:
    pytest.skip("cut_cross_entropy is not installed")
  if not torch.cuda.is_available():
    pytest.skip("cut_cross_entropy default kernel requires CUDA")
  _assert_matches_cross_entropy(
    LossCase(
      name="cut_cross_entropy",
      loss_fn=lambda hidden, weight, targets: cut_cross_entropy(
        hidden,
        weight,
        targets,
      ),
    ),
    device=torch.device("cuda"),
    dtype=torch.float16,
    atol=2e-2,
    rtol=2e-2,
  )


def _assert_matches_cross_entropy(
  case: LossCase,
  *,
  device: torch.device | str = "cpu",
  dtype: torch.dtype = torch.float32,
  atol: float | None = None,
  rtol: float | None = None,
) -> None:
  torch.manual_seed(0)
  hidden = torch.randn(3, 5, 7, device=device, dtype=dtype, requires_grad=True)
  weight = torch.randn(11, 7, device=device, dtype=dtype, requires_grad=True)
  targets = torch.randint(0, 11, (3, 5), device=device)

  expected_hidden = hidden.detach().clone().requires_grad_(True)
  expected_weight = weight.detach().clone().requires_grad_(True)
  expected_loss = F.cross_entropy(
    F.linear(expected_hidden, expected_weight).view(-1, expected_weight.shape[0]),
    targets.view(-1),
  )
  expected_loss.backward()

  actual_hidden = hidden.detach().clone().requires_grad_(True)
  actual_weight = weight.detach().clone().requires_grad_(True)
  actual_loss = case.loss_fn(actual_hidden, actual_weight, targets)
  actual_loss.backward()

  torch.testing.assert_close(
    actual_loss,
    expected_loss,
    atol=atol,
    rtol=rtol,
    check_dtype=False,
  )
  torch.testing.assert_close(
    actual_hidden.grad,
    expected_hidden.grad,
    atol=atol,
    rtol=rtol,
  )
  torch.testing.assert_close(
    actual_weight.grad,
    expected_weight.grad,
    atol=atol,
    rtol=rtol,
  )


def test_language_model_loss_dispatches_linear_cross_entropy() -> None:
  torch.manual_seed(0)
  hidden = torch.randn(2, 4, 6, requires_grad=True)
  weight = torch.randn(9, 6, requires_grad=True)
  targets = torch.randint(0, 9, (2, 4))

  expected = linear_cross_entropy(
    hidden_states=hidden,
    classifier_weight=weight,
    targets=targets,
    chunk_size=3,
  )
  actual = language_model_loss(
    hidden_states=hidden,
    classifier_weight=weight,
    targets=targets,
    backend="linear_cross_entropy",
    chunk_size=3,
  )

  torch.testing.assert_close(actual, expected)


def _cut_cross_entropy():
  try:
    from cut_cross_entropy import linear_cross_entropy as cce_linear_cross_entropy
  except ModuleNotFoundError:
    return None

  def loss_fn(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
  ) -> torch.Tensor:
    try:
      return cce_linear_cross_entropy(hidden, weight, targets)
    except TypeError:
      return cce_linear_cross_entropy(hidden, weight, targets, impl="cce")

  return loss_fn
