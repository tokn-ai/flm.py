"""TensorBoard run sink."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flm_train.config import (
  ExperimentConfig,
  TensorBoardSinkConfig,
  config_to_plain,
)
from flm_train.sinks.base import (
  JsonValue,
  RunContext,
  RunStatus,
  Scalar,
  flatten_json_metrics,
)
from flm_train.types import TrainingResult


class TensorBoardRunSink:
  def __init__(
    self,
    config: TensorBoardSinkConfig,
    *,
    writer: Any | None = None,
  ) -> None:
    self.config = config
    self.writer = writer
    self.system_metrics_step = 0

  def start_run(self, context: RunContext, config: ExperimentConfig) -> None:
    if self.writer is None:
      log_dir = self.config.log_dir or context.run_dir / "tensorboard"
      self.writer = _create_summary_writer(log_dir, flush_secs=self.config.flush_secs)
    self.write_config(config)
    self.log_status("running")

  def write_config(self, config: ExperimentConfig) -> None:
    self._writer().add_text(
      "config/json",
      json.dumps(config_to_plain(config), indent=2, sort_keys=True),
      0,
    )

  def log_status(self, status: RunStatus, message: str | None = None) -> None:
    payload = {"status": status}
    if message is not None:
      payload["message"] = message
    self._writer().add_text("status", json.dumps(payload, sort_keys=True), 0)

  def log_metrics(self, metrics: dict[str, Scalar], step: int) -> None:
    writer = self._writer()
    for name, value in metrics.items():
      if isinstance(value, int | float | bool):
        writer.add_scalar(name, value, step)
      else:
        writer.add_text(name, str(value), step)

  def log_system_metrics(self, metrics: dict[str, JsonValue]) -> None:
    writer = self._writer()
    for name, value in flatten_json_metrics(metrics, prefix="system").items():
      if isinstance(value, int | float | bool):
        writer.add_scalar(name, value, self.system_metrics_step)
      else:
        writer.add_text(name, value, self.system_metrics_step)
    self.system_metrics_step += 1

  def log_artifact(self, path: Path, name: str | None = None) -> None:
    self._writer().add_text(f"artifact/{name or path.name}", str(path), 0)

  def finish_run(self, result: TrainingResult) -> None:
    self._writer().add_text(
      "result/json",
      json.dumps(asdict(result), indent=2, sort_keys=True),
      0,
    )
    self.log_status("success")

  def close(self) -> None:
    if self.writer is not None:
      self.writer.close()

  def _writer(self) -> Any:
    if self.writer is None:
      raise RuntimeError("tensorboard sink has not been started")
    return self.writer


def _create_summary_writer(log_dir: Path, *, flush_secs: int) -> Any:
  try:
    from torch.utils.tensorboard import SummaryWriter
  except ImportError as exc:
    raise RuntimeError(
      "tensorboard sink requires the tensorboard extra; install tensorboard"
    ) from exc
  return SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
