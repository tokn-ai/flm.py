"""Loss helpers for model building blocks."""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

LossBackend = Literal[
  "cross_entropy",
  "linear_cross_entropy",
  "cut_cross_entropy",
  "tilelang_linear_cross_entropy",
]


def language_model_loss(
  *,
  hidden_states: torch.Tensor,
  classifier_weight: torch.Tensor,
  targets: torch.Tensor,
  backend: LossBackend,
  chunk_size: int,
) -> torch.Tensor:
  if backend == "cross_entropy":
    logits = F.linear(hidden_states, classifier_weight)
    return F.cross_entropy(
      logits.view(-1, logits.size(-1)),
      targets.view(-1),
    )
  if backend == "linear_cross_entropy":
    return linear_cross_entropy(
      hidden_states=hidden_states,
      classifier_weight=classifier_weight,
      targets=targets,
      chunk_size=chunk_size,
    )
  if backend == "tilelang_linear_cross_entropy":
    from flm_modules.kernels.tilelang import tilelang_linear_cross_entropy

    return tilelang_linear_cross_entropy(
      hidden_states,
      classifier_weight,
      targets,
    )
  if backend == "cut_cross_entropy":
    try:
      from cut_cross_entropy import linear_cross_entropy as cut_cross_entropy
    except ModuleNotFoundError as exc:
      raise ImportError(
        "cut_cross_entropy backend requires the cut-cross-entropy package"
      ) from exc
    try:
      return cut_cross_entropy(hidden_states, classifier_weight, targets)
    except TypeError:
      return cut_cross_entropy(
        hidden_states,
        classifier_weight,
        targets,
        impl="cce",
      )
  raise ValueError(f"unknown loss backend: {backend}")


def linear_cross_entropy(
  *,
  hidden_states: torch.Tensor,
  classifier_weight: torch.Tensor,
  targets: torch.Tensor,
  chunk_size: int,
) -> torch.Tensor:
  """Torch-only chunked linear cross entropy.

  This avoids materializing the full batch-token-vocab logits tensor. It is a
  compatibility implementation, not the fused Cut Cross-Entropy kernel.
  """
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")
  hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
  labels = targets.reshape(-1)
  total = hidden.new_zeros(())
  for start in range(0, hidden.shape[0], chunk_size):
    end = min(start + chunk_size, hidden.shape[0])
    logits = F.linear(hidden[start:end], classifier_weight)
    total = total + F.cross_entropy(logits, labels[start:end], reduction="sum")
  return total / labels.numel()
