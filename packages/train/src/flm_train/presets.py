"""Preset training workflows used by experiments and smoke tests."""

from __future__ import annotations

from collections.abc import Callable

import torch
from flm_datasets import get_tokenizer
from flm_modules import configure_adamw

from flm_train.data import build_repo_source_dataset
from flm_train.models import build_model
from flm_train.trainer import LanguageModelTrainer, TrainStepMetrics
from flm_train.types import TrainConfig, TrainingResult


def train_on_repo_sources(
  config: TrainConfig,
  *,
  on_step: Callable[[TrainStepMetrics], None] | None = None,
) -> TrainingResult:
  torch.manual_seed(config.seed)

  dataset_bundle = build_repo_source_dataset(config)
  encoding = get_tokenizer(config.encoding_name)
  model = build_model(
    config,
    vocab_size=encoding.n_vocab,
  ).to(config.device)
  optimizer = configure_adamw(
    model,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
  )
  trainer = LanguageModelTrainer(
    model=model,
    optimizer=optimizer,
    dataloader=dataset_bundle.dataloader,
    device=config.device,
    steps=config.steps,
    on_step=on_step,
  )
  step_metrics = trainer.train()

  return TrainingResult(
    losses=[metrics.loss for metrics in step_metrics],
    token_count=dataset_bundle.token_count,
    file_count=dataset_bundle.file_count,
  )
