"""Local filesystem run sink."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from flm_train.config import (
  ExperimentConfig,
  FilesSinkConfig,
  config_to_plain,
  write_yaml,
)
from flm_train.sinks.base import JsonValue, RunContext, RunStatus, Scalar, utc_now
from flm_train.types import TrainingResult


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
      "updated_at": utc_now(),
    }
    if message is not None:
      payload["message"] = message
    _write_json(self._run_dir() / self.config.status_json, payload)

  def log_metrics(self, metrics: dict[str, Scalar], step: int) -> None:
    payload: dict[str, Scalar] = {
      "step": step,
      "time": utc_now(),
      **metrics,
    }
    path = self._run_dir() / self.config.metrics_jsonl
    with path.open("a", encoding="utf-8") as file:
      file.write(json.dumps(payload, sort_keys=True) + "\n")

  def log_system_metrics(self, metrics: dict[str, JsonValue]) -> None:
    payload: dict[str, JsonValue] = {
      "time": utc_now(),
      **metrics,
    }
    path = self._run_dir() / self.config.system_metrics_jsonl
    with path.open("a", encoding="utf-8") as file:
      file.write(json.dumps(payload, sort_keys=True) + "\n")

  def log_artifact(self, path: Path, name: str | None = None) -> None:
    manifest_path = self._run_dir() / "artifacts.jsonl"
    payload = {
      "path": str(path),
      "name": name or path.name,
      "time": utc_now(),
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


def _write_json(path: Path, payload: object) -> None:
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
