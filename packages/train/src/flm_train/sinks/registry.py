"""Run sink construction."""

from __future__ import annotations

from flm_train.config import (
  ExperimentConfig,
  FilesSinkConfig,
  MlflowSinkConfig,
  SinkConfig,
  TensorBoardSinkConfig,
  WandbSinkConfig,
)
from flm_train.sinks.base import CompositeRunSink, RunSink
from flm_train.sinks.files import FilesRunSink
from flm_train.sinks.mlflow import MlflowRunSink
from flm_train.sinks.tensorboard import TensorBoardRunSink
from flm_train.sinks.wandb import WandbRunSink


def build_run_sink(config: ExperimentConfig) -> CompositeRunSink:
  sink_configs = config.sinks or (FilesSinkConfig(),)
  return CompositeRunSink(
    tuple(_build_sink(sink_config) for sink_config in sink_configs)
  )


def _build_sink(config: SinkConfig) -> RunSink:
  if isinstance(config, FilesSinkConfig):
    return FilesRunSink(config)
  if isinstance(config, TensorBoardSinkConfig):
    return TensorBoardRunSink(config)
  if isinstance(config, MlflowSinkConfig):
    return MlflowRunSink(config)
  if isinstance(config, WandbSinkConfig):
    return WandbRunSink(config)
  raise ValueError(f"unsupported sink config: {config}")
