"""Experiment run sinks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from flm_train.config import (
  ExperimentConfig,
  FilesSinkConfig,
  SinkConfig,
  config_to_plain,
  write_yaml,
)
from flm_train.train import TrainingResult

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


class FilesRunSink:
  def __init__(self, config: FilesSinkConfig) -> None:
    self.config = config
    self.run_dir: Path | None = None

  def start_run(self, context: RunContext, config: ExperimentConfig) -> None:
    self.run_dir = self.config.run_dir or context.run_dir
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self.write_config(config)
    self.log_status("running")

  def write_config(self, config: ExperimentConfig) -> None:
    run_dir = self._run_dir()
    _write_json(run_dir / self.config.config_json, config_to_plain(config))
    write_yaml(
      run_dir / self.config.resolved_config_yaml,
      config_to_plain(config),
    )

  def log_status(self, status: RunStatus, message: str | None = None) -> None:
    payload: dict[str, Scalar] = {
      "status": status,
      "updated_at": _utc_now(),
    }
    if message is not None:
      payload["message"] = message
    _write_json(self._run_dir() / self.config.status_json, payload)

  def log_metrics(self, metrics: dict[str, Scalar], step: int) -> None:
    payload: dict[str, Scalar] = {
      "step": step,
      "time": _utc_now(),
      **metrics,
    }
    path = self._run_dir() / self.config.metrics_jsonl
    with path.open("a", encoding="utf-8") as file:
      file.write(json.dumps(payload, sort_keys=True) + "\n")

  def log_artifact(self, path: Path, name: str | None = None) -> None:
    manifest_path = self._run_dir() / "artifacts.jsonl"
    payload = {
      "path": str(path),
      "name": name or path.name,
      "time": _utc_now(),
    }
    with manifest_path.open("a", encoding="utf-8") as file:
      file.write(json.dumps(payload, sort_keys=True) + "\n")

  def finish_run(self, result: TrainingResult) -> None:
    _write_json(self._run_dir() / self.config.result_json, asdict(result))
    self.log_status("success")

  def close(self) -> None:
    return None

  def _run_dir(self) -> Path:
    if self.run_dir is None:
      raise RuntimeError("files sink has not been started")
    return self.run_dir


def build_run_sink(config: ExperimentConfig) -> CompositeRunSink:
  sink_configs = config.sinks or (FilesSinkConfig(),)
  return CompositeRunSink(
    tuple(_build_sink(sink_config) for sink_config in sink_configs)
  )


def _build_sink(config: SinkConfig) -> RunSink:
  if config.kind == "files":
    return FilesRunSink(config)
  raise ValueError(f"unsupported sink kind: {config.kind}")


def _write_json(path: Path, payload: object) -> None:
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _utc_now() -> str:
  return datetime.now(UTC).isoformat()
