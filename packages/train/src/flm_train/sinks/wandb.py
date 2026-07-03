"""Weights & Biases run sink."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from flm_train.config import ExperimentConfig, WandbSinkConfig, config_to_plain
from flm_train.sinks.base import (
  JsonValue,
  RunContext,
  RunStatus,
  Scalar,
  flatten_json_metrics,
)
from flm_train.types import TrainingResult


class WandbRunSink:
  def __init__(self, config: WandbSinkConfig, *, module: Any | None = None) -> None:
    self.config = config
    self.module = module
    self.run: Any | None = None
    self.system_metrics_step = 0

  def start_run(self, context: RunContext, config: ExperimentConfig) -> None:
    module = self._module()
    self.run = module.init(
      project=self.config.project,
      entity=self.config.entity,
      id=config.run.id,
      name=self.config.name or config.run.name or config.name,
      mode=self.config.mode,
      dir=str(self.config.dir or context.run_dir),
      config=config_to_plain(config),
      tags=list(self.config.tags) if self.config.tags else None,
      group=self.config.group or config.run.group or config.name,
      job_type=self.config.job_type,
    )
    self.log_status("running")

  def write_config(self, config: ExperimentConfig) -> None:
    if self.run is not None and hasattr(self.run, "config"):
      self.run.config.update(config_to_plain(config), allow_val_change=True)

  def log_status(self, status: RunStatus, message: str | None = None) -> None:
    payload = {"status": status}
    if message is not None:
      payload["status_message"] = message
    self._module().log(payload)

  def log_metrics(self, metrics: dict[str, Scalar], step: int) -> None:
    self._module().log(metrics, step=step)

  def log_system_metrics(self, metrics: dict[str, JsonValue]) -> None:
    payload = flatten_json_metrics(metrics, prefix="system")
    payload["system/sample"] = self.system_metrics_step
    self._module().log(payload)
    self.system_metrics_step += 1

  def log_artifact(self, path: Path, name: str | None = None) -> None:
    module = self._module()
    artifact = module.Artifact(name or path.name, type="artifact")
    artifact.add_file(str(path))
    module.log_artifact(artifact)

  def finish_run(self, result: TrainingResult) -> None:
    if self.run is not None:
      self.run.summary.update(asdict(result))
    self.log_status("success")
    self.close()

  def close(self) -> None:
    if self.run is not None:
      self.run.finish()
      self.run = None

  def _module(self) -> Any:
    if self.module is None:
      try:
        import wandb
      except ImportError as exc:
        raise RuntimeError(
          "wandb sink requires the wandb extra; install flm-train[wandb]"
        ) from exc
      self.module = wandb
    return self.module
