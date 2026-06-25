"""Training workflows."""

__all__ = [
  "TrainConfig",
  "TrainingResult",
  "train_on_repo_sources",
]


def __getattr__(name: str):
  if name in __all__:
    from flm_train.train import TrainConfig, TrainingResult, train_on_repo_sources

    exports = {
      "TrainConfig": TrainConfig,
      "TrainingResult": TrainingResult,
      "train_on_repo_sources": train_on_repo_sources,
    }
    return exports[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
