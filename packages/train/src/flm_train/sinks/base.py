"""Shared run sink contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from flm_train.config import ExperimentConfig
from flm_train.types import TrainingResult

Scalar = float | int | bool | str
RunStatus = Literal["running", "success", "failed"]


@dataclass(frozen=True)
class RunContext:
  run_dir: Path


class RunSink(Protocol):
  def start_run(self, context: RunContext, config: ExperimentConfig) -> None: ...

  def write_config(self, config: ExperimentConfig) -> None: ...

  def log_status(self, status: RunStatus, message: str | None = None) -> None: ...

  def log_metrics(self, metrics: dict[str, Scalar], step: int) -> None: ...

  def log_artifact(self, path: Path, name: str | None = None) -> None: ...

  def finish_run(self, result: TrainingResult) -> None: ...

  def close(self) -> None: ...


class CompositeRunSink:
  def __init__(self, sinks: tuple[RunSink, ...]) -> None:
    self.sinks = sinks

  def start_run(self, context: RunContext, config: ExperimentConfig) -> None:
    for sink in self.sinks:
      sink.start_run(context, config)

  def write_config(self, config: ExperimentConfig) -> None:
    for sink in self.sinks:
      sink.write_config(config)

  def log_status(self, status: RunStatus, message: str | None = None) -> None:
    for sink in self.sinks:
      sink.log_status(status, message)

  def log_metrics(self, metrics: dict[str, Scalar], step: int) -> None:
    for sink in self.sinks:
      sink.log_metrics(metrics, step)

  def log_artifact(self, path: Path, name: str | None = None) -> None:
    for sink in self.sinks:
      sink.log_artifact(path, name)

  def finish_run(self, result: TrainingResult) -> None:
    for sink in self.sinks:
      sink.finish_run(result)

  def close(self) -> None:
    for sink in reversed(self.sinks):
      sink.close()


def utc_now() -> str:
  return datetime.now(UTC).isoformat()
