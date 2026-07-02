"""Experiment execution loop and run artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from flm_train.config import ExperimentConfig, config_to_plain, write_yaml
from flm_train.train import TrainingResult, train_on_repo_sources

LogFn = Callable[[str], None]


class ExperimentRunner:
  def __init__(self, config: ExperimentConfig, log: LogFn | None = None) -> None:
    self.config = config
    self.log = log
    self.run_dir = config.run_dir

  def run(self) -> TrainingResult:
    self.prepare_run_dir()
    self.write_resolved_config()
    self._log(f"run_dir={self.run_dir}")

    result = self.train()
    self.write_result(result)
    self.report_result(result)
    return result

  def prepare_run_dir(self) -> None:
    self.run_dir.mkdir(parents=True, exist_ok=True)

  def write_resolved_config(self) -> None:
    write_yaml(
      self.run_dir / "config.resolved.yaml",
      config_to_plain(self.config),
    )

  def train(self) -> TrainingResult:
    return train_on_repo_sources(self.config.to_train_config())

  def write_result(self, result: TrainingResult) -> None:
    (self.run_dir / "result.json").write_text(
      json.dumps(asdict(result), indent=2) + "\n",
      encoding="utf-8",
    )

  def report_result(self, result: TrainingResult) -> None:
    for step, loss in enumerate(result.losses, start=1):
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
