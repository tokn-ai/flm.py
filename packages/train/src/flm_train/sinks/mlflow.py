"""MLflow run sink."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from flm_train.config import ExperimentConfig, MlflowSinkConfig, config_to_plain
from flm_train.sinks.base import RunContext, RunStatus, Scalar
from flm_train.types import TrainingResult


class MlflowRunSink:
  def __init__(self, config: MlflowSinkConfig, *, client: Any | None = None) -> None:
    self.config = config
    self.client = client
    self.active = False

  def start_run(self, context: RunContext, config: ExperimentConfig) -> None:
    del context
    client = self._client()
    if self.config.tracking_uri is not None:
      client.set_tracking_uri(self.config.tracking_uri)
    client.set_experiment(self.config.experiment_name)
    client.start_run(
      run_name=self.config.run_name or config.name,
      nested=self.config.nested,
    )
    self.active = True
    self.write_config(config)
    self.log_status("running")

  def write_config(self, config: ExperimentConfig) -> None:
    for name, value in _flatten(config_to_plain(config)).items():
      self._client().log_param(name, value)

  def log_status(self, status: RunStatus, message: str | None = None) -> None:
    tags = {"status": status}
    if message is not None:
      tags["status_message"] = message
    self._client().set_tags(tags)

  def log_metrics(self, metrics: dict[str, Scalar], step: int) -> None:
    numeric_metrics = {
      name: float(value)
      for name, value in metrics.items()
      if isinstance(value, int | float | bool)
    }
    if numeric_metrics:
      self._client().log_metrics(numeric_metrics, step=step)

  def log_artifact(self, path: Path, name: str | None = None) -> None:
    self._client().log_artifact(str(path), artifact_path=name)

  def finish_run(self, result: TrainingResult) -> None:
    self._client().log_dict(asdict(result), "result.json")
    self.log_status("success")
    self._client().end_run(status="FINISHED")
    self.active = False

  def close(self) -> None:
    if self.active:
      self._client().end_run()
      self.active = False

  def _client(self) -> Any:
    if self.client is None:
      try:
        import mlflow
      except ImportError as exc:
        raise RuntimeError(
          "mlflow sink requires the mlflow extra; install flm-train[mlflow]"
        ) from exc
      self.client = mlflow
    return self.client


def _flatten(value: object, prefix: str = "") -> dict[str, str]:
  if isinstance(value, dict):
    flattened: dict[str, str] = {}
    for key, item in value.items():
      name = f"{prefix}.{key}" if prefix else str(key)
      flattened.update(_flatten(item, name))
    return flattened
  if isinstance(value, list):
    return {prefix: ",".join(str(item) for item in value)}
  return {prefix: str(value)}
