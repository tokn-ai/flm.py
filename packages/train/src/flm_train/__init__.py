"""Training workflows."""

__all__ = [
  "ExperimentConfig",
  "TrainConfig",
  "TrainStepMetrics",
  "TrainingResult",
  "load_experiment_config",
  "run_experiment",
  "train_on_repo_sources",
]


def __getattr__(name: str):
  if name in {"TrainConfig", "TrainingResult", "train_on_repo_sources"}:
    from flm_train.train import TrainConfig, TrainingResult, train_on_repo_sources

    exports = {
      "TrainConfig": TrainConfig,
      "TrainingResult": TrainingResult,
      "train_on_repo_sources": train_on_repo_sources,
    }
    return exports[name]
  if name in {"TrainStepMetrics"}:
    from flm_train.trainer import TrainStepMetrics

    exports = {
      "TrainStepMetrics": TrainStepMetrics,
    }
    return exports[name]
  if name in {"ExperimentConfig", "load_experiment_config", "run_experiment"}:
    from flm_train.experiment import (
      ExperimentConfig,
      load_experiment_config,
      run_experiment,
    )

    exports = {
      "ExperimentConfig": ExperimentConfig,
      "load_experiment_config": load_experiment_config,
      "run_experiment": run_experiment,
    }
    return exports[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
