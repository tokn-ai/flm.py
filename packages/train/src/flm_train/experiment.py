"""Experiment compatibility exports."""

from __future__ import annotations

from flm_train.cli import build_parser, main, parse_args, run_from_args
from flm_train.config import (
  DataConfig,
  ExperimentConfig,
  ExperimentOverrides,
  FilesSinkConfig,
  ModelConfig,
  OptimizerConfig,
  OutputConfig,
  RunTrainConfig,
  SinkConfig,
  apply_overrides,
  config_to_plain,
  load_experiment_config,
  parse_experiment_config,
  write_yaml,
)
from flm_train.runner import ExperimentRunner, run_experiment
from flm_train.sinks import (
  CompositeRunSink,
  FilesRunSink,
  RunContext,
  RunSink,
  build_run_sink,
)

__all__ = [
  "DataConfig",
  "ExperimentConfig",
  "ExperimentOverrides",
  "ExperimentRunner",
  "FilesRunSink",
  "FilesSinkConfig",
  "CompositeRunSink",
  "ModelConfig",
  "OptimizerConfig",
  "OutputConfig",
  "RunContext",
  "RunSink",
  "RunTrainConfig",
  "SinkConfig",
  "apply_overrides",
  "build_parser",
  "build_run_sink",
  "config_to_plain",
  "load_experiment_config",
  "main",
  "parse_args",
  "parse_experiment_config",
  "run_experiment",
  "run_from_args",
  "write_yaml",
]


if __name__ == "__main__":
  main()
