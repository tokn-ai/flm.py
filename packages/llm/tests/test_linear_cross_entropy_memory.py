from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from flm_llm.losses import linear_cross_entropy
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryCase:
  name: str
  loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def test_linear_cross_entropy_backward_memory_spike() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA memory spike test requires CUDA")

  device = torch.device("cuda")
  dtype = torch.float16
  token_count = 4 * 256
  d_model = 512
  vocab_size = 8192
  torch.manual_seed(0)
  hidden = torch.randn(token_count, d_model, device=device, dtype=dtype)
  weight = torch.randn(vocab_size, d_model, device=device, dtype=dtype)
  targets = torch.randint(0, vocab_size, (token_count,), device=device)

  cases = _memory_cases(chunk_size=64)
  peaks = {
    case.name: _measure_backward_peak(
      case.loss_fn,
      hidden=hidden,
      weight=weight,
      targets=targets,
    )
    for case in cases
  }

  print(f"linear_cross_entropy backward memory peaks: {peaks}")
  assert peaks["linear+cross_entropy"] > peaks["chunked_cce"]
  if "F.linear_cross_entropy" in peaks:
    assert peaks["linear+cross_entropy"] >= peaks["F.linear_cross_entropy"]
  if "cut_cross_entropy" in peaks:
    assert peaks["linear+cross_entropy"] >= peaks["cut_cross_entropy"]


def _memory_cases(*, chunk_size: int) -> list[MemoryCase]:
  cases = [
    MemoryCase(
      name="linear+cross_entropy",
      loss_fn=lambda hidden, weight, targets: F.cross_entropy(
        F.linear(hidden, weight),
        targets,
      ),
    ),
    MemoryCase(
      name="chunked_cce",
      loss_fn=lambda hidden, weight, targets: linear_cross_entropy(
        hidden_states=hidden,
        classifier_weight=weight,
        targets=targets,
        chunk_size=chunk_size,
      ),
    ),
  ]
  torch_linear_cross_entropy = getattr(F, "linear_cross_entropy", None)
  if torch_linear_cross_entropy is not None:
    cases.append(
      MemoryCase(
        name="F.linear_cross_entropy",
        loss_fn=lambda hidden, weight, targets: torch_linear_cross_entropy(
          hidden,
          weight,
          targets,
        ),
      )
    )
  cut_cross_entropy = _cut_cross_entropy()
  if cut_cross_entropy is not None:
    cases.append(
      MemoryCase(
        name="cut_cross_entropy",
        loss_fn=lambda hidden, weight, targets: cut_cross_entropy(
          hidden,
          weight,
          targets,
        ),
      )
    )
  return cases


def _measure_backward_peak(
  loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
  *,
  hidden: torch.Tensor,
  weight: torch.Tensor,
  targets: torch.Tensor,
) -> int:
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  start_allocated = torch.cuda.memory_allocated()
  measured_hidden = hidden.detach().clone().requires_grad_(True)
  measured_weight = weight.detach().clone().requires_grad_(True)
  loss = loss_fn(measured_hidden, measured_weight, targets)
  loss.backward()
  torch.cuda.synchronize()
  return int(torch.cuda.max_memory_allocated() - start_allocated)


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
