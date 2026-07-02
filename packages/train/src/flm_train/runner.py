"""Experiment execution loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from flm_train.config import ExperimentConfig
from flm_train.sinks import RunContext, build_run_sink
from flm_train.train import TrainingResult, train_on_repo_sources
from flm_train.trainer import TrainStepMetrics

LogFn = Callable[[str], None]


class ExperimentRunner:
  def __init__(self, config: ExperimentConfig, log: LogFn | None = None) -> None:
    self.config = config
    self.log = log
    self.run_dir = config.run_dir

  def run(self) -> TrainingResult:
    sink = build_run_sink(self.config)
    context = RunContext(run_dir=self.run_dir)
    self._log(f"run_dir={self.run_dir}")
    try:
      sink.start_run(context, self.config)
      result = self.train(on_step=lambda metrics: self.report_step(metrics, sink=sink))
      self.report_result(result)
      sink.finish_run(result)
      return result
    except Exception as exc:
      sink.log_status("failed", message=str(exc))
      raise
    finally:
      sink.close()

  def train(
    self,
    *,
    on_step,
  ) -> TrainingResult:
    return train_on_repo_sources(self.config.to_train_config(), on_step=on_step)

  def report_step(self, metrics: TrainStepMetrics, sink) -> None:
    sink.log_metrics(metrics.to_log_dict(), step=metrics.step)
    self._log(f"step={metrics.step} loss={metrics.loss:.4f}")

  def report_result(self, result: TrainingResult) -> None:
    self._log(f"tokens={result.token_count} files={result.file_count}")

  def _log(self, message: str) -> None:
    if self.log is not None:
      self.log(message)


def run_experiment(
  config: ExperimentConfig,
  *,
  log: LogFn | None = None,
) -> TrainingResult:
  return ExperimentRunner(config, log=log).run()


def result_path(run_dir: Path) -> Path:
  return run_dir / "result.json"
