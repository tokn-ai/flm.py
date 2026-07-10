"""Reusable language-model training loop."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader

from flm_train.checkpoints import (
  CheckpointState,
  latest_checkpoint_path,
  load_checkpoint,
  prune_checkpoints,
  save_checkpoint,
)
from flm_train.schedules import OptimizerSchedule
from flm_train.types import CheckpointConfig


class LanguageModel(Protocol):
  def train(self, mode: bool = True): ...

  def state_dict(self): ...

  def load_state_dict(self, state_dict): ...

  def parameters(self) -> Iterable[torch.nn.Parameter]: ...

  def __call__(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    return_logits: bool = True,
  ) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...


@dataclass(frozen=True)
class TrainStepMetrics:
  step: int
  loss: float
  learning_rate: float
  tokens: int
  tokens_seen: int
  grad_norm: float
  bits_per_byte: float
  step_time_sec: float
  tokens_per_sec: float

  def to_log_dict(self) -> dict[str, float | int]:
    return {
      "train/loss": self.loss,
      "train/lr": self.learning_rate,
      "train/tokens": self.tokens,
      "train/tokens_seen": self.tokens_seen,
      "train/grad_norm": self.grad_norm,
      "train/bpb": self.bits_per_byte,
      "system/step_time_sec": self.step_time_sec,
      "train/tokens_per_sec": self.tokens_per_sec,
    }


@dataclass(frozen=True)
class EvalMetrics:
  step: int
  split: str
  loss: float
  bits_per_byte: float
  tokens: int

  def to_log_dict(self) -> dict[str, float | int]:
    return {
      f"eval/{self.split}_loss": self.loss,
      f"eval/{self.split}_bpb": self.bits_per_byte,
      f"eval/{self.split}_tokens": self.tokens,
    }


@dataclass(frozen=True)
class RolloutSample:
  name: str
  prompt: str
  prompt_tokens: tuple[int, ...]
  prompt_log_probs: tuple[float, ...]
  tokens: tuple[int, ...]
  token_texts: tuple[str, ...]
  log_probs: tuple[float, ...]
  entropy: tuple[float, ...]
  top_tokens: tuple[tuple[int, ...], ...]
  top_token_texts: tuple[tuple[str, ...], ...]
  top_log_probs: tuple[tuple[float, ...], ...]
  text: str


@dataclass(frozen=True)
class RolloutBatch:
  step: int
  samples: tuple[RolloutSample, ...]


StepCallback = Callable[[TrainStepMetrics], None]
EvalCallback = Callable[[int, LanguageModel], EvalMetrics]
EvalMetricsCallback = Callable[[EvalMetrics], None]
RolloutCallback = Callable[[int, LanguageModel], RolloutBatch]
RolloutBatchCallback = Callable[[RolloutBatch], None]
CheckpointCallback = Callable[[Path, int], None]


class LanguageModelTrainer:
  def __init__(
    self,
    *,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    optimizer_schedule: OptimizerSchedule | None = None,
    dataloader: DataLoader,
    device: str,
    steps: int,
    bytes_per_token: float,
    max_grad_norm: float | None = 1.0,
    on_step: StepCallback | None = None,
    eval_every_steps: int | None = None,
    evaluate: EvalCallback | None = None,
    on_eval: EvalMetricsCallback | None = None,
    rollout_every_steps: int | None = None,
    rollout: RolloutCallback | None = None,
    on_rollout: RolloutBatchCallback | None = None,
    checkpoint: CheckpointConfig | None = None,
    checkpoint_dir: Path | None = None,
    on_checkpoint: CheckpointCallback | None = None,
  ) -> None:
    if steps < 0:
      raise ValueError("steps must be non-negative")
    self.model = model
    self.optimizer = optimizer
    self.optimizer_schedule = optimizer_schedule
    self.dataloader = dataloader
    self.device = device
    self.steps = steps
    self.bytes_per_token = bytes_per_token
    self.max_grad_norm = max_grad_norm
    self.on_step = on_step
    self.eval_every_steps = eval_every_steps
    self.evaluate = evaluate
    self.on_eval = on_eval
    self.rollout_every_steps = rollout_every_steps
    self.rollout = rollout
    self.on_rollout = on_rollout
    self.checkpoint = checkpoint or CheckpointConfig()
    self.checkpoint_dir = checkpoint_dir
    self.on_checkpoint = on_checkpoint

  def train(self) -> list[TrainStepMetrics]:
    metrics: list[TrainStepMetrics] = []
    iterator = iter(self.dataloader)
    checkpoint_state = self._load_checkpoint_if_requested()
    tokens_seen = checkpoint_state.tokens_seen
    self.model.train()

    for step in range(checkpoint_state.step + 1, self.steps + 1):
      if self.optimizer_schedule is not None:
        self.optimizer_schedule.apply(step)
      input_ids, targets, iterator = self._next_batch(iterator)
      input_ids = input_ids.to(self.device)
      targets = targets.to(self.device)

      started_at = time.perf_counter()
      self.optimizer.zero_grad(set_to_none=True)
      _, loss = self.model(input_ids, targets, return_logits=False)
      if loss is None:
        raise RuntimeError("training loss was not produced")
      loss.backward()
      grad_norm = _clip_or_measure_grad_norm(
        self.model.parameters(),
        self.max_grad_norm,
      )
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
        grad_norm=grad_norm,
        bits_per_byte=_loss_to_bits_per_byte(
          loss=float(loss.detach().cpu()),
          bytes_per_token=self.bytes_per_token,
        ),
        step_time_sec=step_time_sec,
        tokens_per_sec=token_count / max(step_time_sec, 1e-12),
      )
      metrics.append(step_metrics)
      if self.on_step is not None:
        self.on_step(step_metrics)
      if self._should_eval(step) and self.evaluate is not None:
        eval_metrics = self.evaluate(step, self.model)
        if self.on_eval is not None:
          self.on_eval(eval_metrics)
        self.model.train()
      if self._should_rollout(step) and self.rollout is not None:
        rollout_batch = self.rollout(step, self.model)
        if self.on_rollout is not None:
          self.on_rollout(rollout_batch)
        self.model.train()
      if self._should_checkpoint(step):
        path = self._save_checkpoint(step=step, tokens_seen=tokens_seen)
        if self.on_checkpoint is not None:
          self.on_checkpoint(path, step)

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

  def _should_eval(self, step: int) -> bool:
    return (
      self.eval_every_steps is not None
      and self.eval_every_steps > 0
      and step % self.eval_every_steps == 0
    )

  def _should_rollout(self, step: int) -> bool:
    return (
      self.rollout_every_steps is not None
      and self.rollout_every_steps > 0
      and step % self.rollout_every_steps == 0
    )

  def _should_checkpoint(self, step: int) -> bool:
    return (
      self.checkpoint.enabled
      and self.checkpoint_dir is not None
      and self.checkpoint.every_steps > 0
      and step % self.checkpoint.every_steps == 0
    )

  def _save_checkpoint(self, *, step: int, tokens_seen: int) -> Path:
    if self.checkpoint_dir is None:
      raise RuntimeError("checkpoint_dir is required to save checkpoints")
    path = save_checkpoint(
      checkpoint_dir=self.checkpoint_dir,
      model=self.model,
      optimizer=self.optimizer,
      state=CheckpointState(step=step, tokens_seen=tokens_seen),
    )
    prune_checkpoints(
      checkpoint_dir=self.checkpoint_dir,
      keep_last=self.checkpoint.keep_last,
    )
    return path

  def _load_checkpoint_if_requested(self) -> CheckpointState:
    if (
      not self.checkpoint.enabled
      or self.checkpoint.resume is None
      or self.checkpoint.resume == "none"
    ):
      return CheckpointState(step=0, tokens_seen=0)
    if self.checkpoint_dir is None:
      raise RuntimeError("checkpoint_dir is required to resume checkpoints")
    if self.checkpoint.resume == "auto":
      path = latest_checkpoint_path(self.checkpoint_dir)
      if path is None:
        return CheckpointState(step=0, tokens_seen=0)
    else:
      path = Path(self.checkpoint.resume)
    return load_checkpoint(
      path=path,
      model=self.model,
      optimizer=self.optimizer,
      map_location=self.device,
    )


def _learning_rate(optimizer: torch.optim.Optimizer) -> float:
  if not optimizer.param_groups:
    return 0.0
  return float(optimizer.param_groups[0].get("lr", 0.0))


def _clip_or_measure_grad_norm(
  parameters: Iterable[torch.nn.Parameter],
  max_grad_norm: float | None,
) -> float:
  params = [param for param in parameters if param.grad is not None]
  if not params:
    return 0.0
  if max_grad_norm is not None:
    return float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
  return float(
    torch.linalg.vector_norm(
      torch.stack(
        [torch.linalg.vector_norm(param.grad.detach(), ord=2) for param in params]
      ),
      ord=2,
    )
  )


def _loss_to_bits_per_byte(*, loss: float, bytes_per_token: float) -> float:
  if bytes_per_token <= 0:
    return 0.0
  return loss / log(2.0) / bytes_per_token
