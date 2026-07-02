"""Reusable language-model training loop."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.data import DataLoader


class LanguageModel(Protocol):
  def train(self, mode: bool = True): ...

  def __call__(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


@dataclass(frozen=True)
class TrainStepMetrics:
  step: int
  loss: float
  learning_rate: float
  tokens: int
  tokens_seen: int
  step_time_sec: float
  tokens_per_sec: float

  def to_log_dict(self) -> dict[str, float | int]:
    return {
      "train/loss": self.loss,
      "train/lr": self.learning_rate,
      "train/tokens": self.tokens,
      "train/tokens_seen": self.tokens_seen,
      "system/step_time_sec": self.step_time_sec,
      "train/tokens_per_sec": self.tokens_per_sec,
    }


StepCallback = Callable[[TrainStepMetrics], None]


class LanguageModelTrainer:
  def __init__(
    self,
    *,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    steps: int,
    on_step: StepCallback | None = None,
  ) -> None:
    if steps < 0:
      raise ValueError("steps must be non-negative")
    self.model = model
    self.optimizer = optimizer
    self.dataloader = dataloader
    self.device = device
    self.steps = steps
    self.on_step = on_step

  def train(self) -> list[TrainStepMetrics]:
    metrics: list[TrainStepMetrics] = []
    iterator = iter(self.dataloader)
    tokens_seen = 0
    self.model.train()

    for step in range(1, self.steps + 1):
      input_ids, targets, iterator = self._next_batch(iterator)
      input_ids = input_ids.to(self.device)
      targets = targets.to(self.device)

      started_at = time.perf_counter()
      self.optimizer.zero_grad(set_to_none=True)
      _, loss = self.model(input_ids, targets)
      if loss is None:
        raise RuntimeError("training loss was not produced")
      loss.backward()
      self.optimizer.step()
      step_time_sec = time.perf_counter() - started_at

      token_count = int(input_ids.numel())
      tokens_seen += token_count
      step_metrics = TrainStepMetrics(
        step=step,
        loss=float(loss.detach().cpu()),
        learning_rate=_learning_rate(self.optimizer),
        tokens=token_count,
        tokens_seen=tokens_seen,
        step_time_sec=step_time_sec,
        tokens_per_sec=token_count / max(step_time_sec, 1e-12),
      )
      metrics.append(step_metrics)
      if self.on_step is not None:
        self.on_step(step_metrics)

    return metrics

  def _next_batch(
    self,
    iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
  ) -> tuple[torch.Tensor, torch.Tensor, Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    try:
      input_ids, targets = next(iterator)
    except StopIteration:
      iterator = iter(self.dataloader)
      input_ids, targets = next(iterator)
    return input_ids, targets, iterator


def _learning_rate(optimizer: torch.optim.Optimizer) -> float:
  if not optimizer.param_groups:
    return 0.0
  return float(optimizer.param_groups[0].get("lr", 0.0))
