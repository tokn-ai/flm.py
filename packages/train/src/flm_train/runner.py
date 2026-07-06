"""Experiment execution loop."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from flm_train.config import ExperimentConfig, RunConfig
from flm_train.data import resolve_data_config
from flm_train.presets import train_language_model
from flm_train.secrets import apply_secret_env, load_secret_env
from flm_train.sinks import RunContext, build_run_sink
from flm_train.svd import checkpoint_ffn_down_svd_metrics
from flm_train.system_metrics import SystemMetricsSampler
from flm_train.trainer import EvalMetrics, RolloutBatch, TrainStepMetrics
from flm_train.types import TrainingResult

LogFn = Callable[[str], None]


class ExperimentRunner:
  def __init__(self, config: ExperimentConfig, log: LogFn | None = None) -> None:
    self.config = config
    self.log = log
    self.run_dir = config.run_dir

  def run(self) -> TrainingResult:
    apply_secret_env(load_secret_env(self.config.secrets.env_file))
    self.config = self.resolved_config()
    self.run_dir = self.config.run_dir
    sink = build_run_sink(self.config)
    context = RunContext(run_dir=self.run_dir)
    self._log(f"run_dir={self.run_dir}")
    sampler = self._system_metrics_sampler(sink)
    try:
      sink.start_run(context, self.config)
      if sampler is not None:
        sampler.start()
      result = self.train(
        on_step=lambda metrics: self.report_step(metrics, sink=sink),
        on_eval=lambda metrics: self.report_eval(metrics, sink=sink),
        on_rollout=lambda batch: self.report_rollout(batch, sink=sink),
        on_checkpoint=lambda path, step: self.report_checkpoint(
          path,
          step=step,
          sink=sink,
        ),
      )
      self.report_result(result)
      sink.finish_run(result)
      return result
    except Exception as exc:
      sink.log_status("failed", message=str(exc))
      raise
    finally:
      if sampler is not None:
        sampler.stop()
      sink.close()

  def train(
    self,
    *,
    on_step,
    on_eval,
    on_rollout,
    on_checkpoint,
  ) -> TrainingResult:
    return train_language_model(
      self.config.to_train_config(),
      on_step=on_step,
      on_eval=on_eval,
      on_rollout=on_rollout,
      checkpoint_dir=self.run_dir / "checkpoints",
      on_checkpoint=on_checkpoint,
    )

  def resolved_config(self) -> ExperimentConfig:
    return ExperimentConfig(
      name=self.config.name,
      data=resolve_data_config(self.config.data),
      model=self.config.model,
      optimizer=self.config.optimizer,
      loop=self.config.loop,
      eval=self.config.eval,
      rollout=self.config.rollout,
      checkpoint=self.config.checkpoint,
      system_metrics=self.config.system_metrics,
      run=resolve_run_config(self.config.name, self.config.run),
      secrets=self.config.secrets,
      output=self.config.output,
      sinks=self.config.sinks,
    )

  def report_step(self, metrics: TrainStepMetrics, sink) -> None:
    sink.log_metrics(metrics.to_log_dict(), step=metrics.step)
    self._log(
      f"step={metrics.step} loss={metrics.loss:.4f} grad_norm={metrics.grad_norm:.4f}"
    )

  def report_eval(self, metrics: EvalMetrics, sink) -> None:
    sink.log_metrics(metrics.to_log_dict(), step=metrics.step)
    self._log(f"step={metrics.step} {metrics.split}_loss={metrics.loss:.4f}")

  def report_rollout(self, batch: RolloutBatch, sink) -> None:
    rollout_dir = self.run_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    path = rollout_dir / f"step-{batch.step:08d}.json"
    path.write_text(
      json.dumps(asdict(batch), indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    sink.log_artifact(path, name=f"rollouts/step-{batch.step:08d}.json")
    self._log(f"step={batch.step} rollouts={path}")

  def report_checkpoint(self, path: Path, *, step: int, sink) -> None:
    sink.log_artifact(path, name=f"checkpoints/step-{step:08d}")
    svd_metrics = checkpoint_ffn_down_svd_metrics(path)
    if svd_metrics:
      sink.log_metrics(svd_metrics, step=step)
    self._log(f"step={step} checkpoint={path}")

  def report_result(self, result: TrainingResult) -> None:
    self._log(
      f"tokens={result.token_count} files={result.file_count} bytes={result.byte_count}"
    )

  def _log(self, message: str) -> None:
    if self.log is not None:
      self.log(message)

  def _system_metrics_sampler(self, sink) -> SystemMetricsSampler | None:
    if not self.config.system_metrics.enabled:
      return None
    return SystemMetricsSampler(
      every_seconds=self.config.system_metrics.every_seconds,
      emit=sink.log_system_metrics,
    )


def run_experiment(
  config: ExperimentConfig,
  *,
  log: LogFn | None = None,
) -> TrainingResult:
  return ExperimentRunner(config, log=log).run()


def result_path(run_dir: Path) -> Path:
  return run_dir / "result.json"


def resolve_run_config(experiment_name: str, run: RunConfig) -> RunConfig:
  run_id = run.id or generate_run_id()
  run_name = run.name or generate_run_name(experiment_name)
  return RunConfig(id=run_id, name=run_name, group=run.group or experiment_name)


def generate_run_id() -> str:
  timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
  return f"{timestamp}-{uuid4().hex[:6]}"


def generate_run_name(experiment_name: str) -> str:
  return f"{experiment_name} {random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"


_ADJECTIVES = (
  "brisk",
  "calm",
  "clear",
  "direct",
  "fresh",
  "quiet",
  "sharp",
  "steady",
)

_NOUNS = (
  "arc",
  "beam",
  "field",
  "forge",
  "signal",
  "spark",
  "trail",
  "vector",
)
