"""Experiment execution loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from flm_train.config import ExperimentConfig
from flm_train.sinks import RunContext, build_run_sink
from flm_train.train import TrainingResult, train_on_repo_sources

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
      result = self.train()
      self.report_result(result, sink=sink)
      sink.finish_run(result)
      return result
    except Exception as exc:
      sink.log_status("failed", message=str(exc))
      raise
    finally:
      sink.close()

  def train(self) -> TrainingResult:
    return train_on_repo_sources(self.config.to_train_config())

  def report_result(self, result: TrainingResult, sink) -> None:
    for step, loss in enumerate(result.losses, start=1):
      sink.log_metrics({"train/loss": loss}, step=step)
      self._log(f"step={step} loss={loss:.4f}")
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
