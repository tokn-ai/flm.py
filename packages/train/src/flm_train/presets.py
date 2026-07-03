"""Preset training workflows used by experiments and smoke tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch
from flm_datasets import get_tokenizer
from flm_modules import configure_adamw

from flm_train.data import build_training_dataset
from flm_train.models import build_model
from flm_train.trainer import (
  EvalMetrics,
  LanguageModel,
  LanguageModelTrainer,
  RolloutBatch,
  RolloutSample,
  TrainStepMetrics,
)
from flm_train.types import RolloutPromptConfig, TrainConfig, TrainingResult


def train_language_model(
  config: TrainConfig,
  *,
  on_step: Callable[[TrainStepMetrics], None] | None = None,
  on_eval: Callable[[EvalMetrics], None] | None = None,
  on_rollout: Callable[[RolloutBatch], None] | None = None,
) -> TrainingResult:
  torch.manual_seed(config.loop.seed)

  dataset_bundle = build_training_dataset(config)
  eval_bundle = None
  if config.eval is not None:
    eval_config = replace(
      config,
      data=replace(config.data, split=config.eval.split),
    )
    eval_bundle = build_training_dataset(eval_config)
  encoding = get_tokenizer(config.data.encoding_name)
  model = build_model(
    config,
    vocab_size=encoding.n_vocab,
  ).to(config.loop.device)
  optimizer = configure_adamw(
    model,
    learning_rate=config.optimizer.learning_rate,
    weight_decay=config.optimizer.weight_decay,
  )
  trainer = LanguageModelTrainer(
    model=model,
    optimizer=optimizer,
    dataloader=dataset_bundle.dataloader,
    device=config.loop.device,
    steps=config.loop.steps,
    on_step=on_step,
    eval_every_steps=config.eval.every_steps if config.eval is not None else None,
    evaluate=None
    if config.eval is None or eval_bundle is None
    else lambda step, model: evaluate_language_model(
      model=model,
      dataloader=eval_bundle.dataloader,
      device=config.loop.device,
      split=config.eval.split,
      max_batches=config.eval.max_batches,
      step=step,
    ),
    on_eval=on_eval,
    rollout_every_steps=config.rollout.every_steps
    if config.rollout is not None and config.rollout.prompts
    else None,
    rollout=None
    if config.rollout is None or not config.rollout.prompts
    else lambda step, model: generate_rollouts(
      model=model,
      prompts=config.rollout.prompts,
      encoding=encoding,
      device=config.loop.device,
      max_seq_len=config.data.seq_len,
      max_new_tokens=config.rollout.max_new_tokens,
      step=step,
    ),
    on_rollout=on_rollout,
  )
  step_metrics = trainer.train()

  return TrainingResult(
    losses=[metrics.loss for metrics in step_metrics],
    token_count=dataset_bundle.token_count,
    file_count=dataset_bundle.file_count,
  )


def evaluate_language_model(
  *,
  model: LanguageModel,
  dataloader,
  device: str,
  split: str,
  max_batches: int,
  step: int,
) -> EvalMetrics:
  if max_batches < 1:
    raise ValueError("max_batches must be positive")
  model.train(False)
  total_loss = 0.0
  total_tokens = 0
  total_batches = 0
  with torch.no_grad():
    for input_ids, targets in dataloader:
      input_ids = input_ids.to(device)
      targets = targets.to(device)
      _, loss = model(input_ids, targets)
      if loss is None:
        raise RuntimeError("eval loss was not produced")
      token_count = int(input_ids.numel())
      total_loss += float(loss.detach().cpu()) * token_count
      total_tokens += token_count
      total_batches += 1
      if total_batches >= max_batches:
        break
  if total_tokens == 0:
    raise RuntimeError("eval dataloader produced no tokens")
  loss = total_loss / total_tokens
  return EvalMetrics(
    step=step,
    split=split,
    loss=loss,
    tokens=total_tokens,
  )


def generate_rollouts(
  *,
  model: LanguageModel,
  prompts: tuple[RolloutPromptConfig, ...],
  encoding,
  device: str,
  max_seq_len: int,
  max_new_tokens: int,
  step: int,
) -> RolloutBatch:
  model.train(False)
  samples = []
  with torch.no_grad():
    for prompt in prompts:
      prompt_tokens = encoding.encode_ordinary(prompt.prompt)
      output_tokens = greedy_decode(
        model=model,
        prompt_tokens=prompt_tokens,
        device=device,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
      )
      text = encoding.decode(output_tokens)
      samples.append(
        RolloutSample(
          name=prompt.name,
          prompt=prompt.prompt,
          completion=encoding.decode(output_tokens[len(prompt_tokens) :]),
          text=text,
        )
      )
  return RolloutBatch(step=step, samples=tuple(samples))


def greedy_decode(
  *,
  model: LanguageModel,
  prompt_tokens: list[int],
  device: str,
  max_seq_len: int,
  max_new_tokens: int,
) -> list[int]:
  if not prompt_tokens:
    raise ValueError("rollout prompt must not be empty")
  tokens = list(prompt_tokens)
  for _ in range(max_new_tokens):
    window = tokens[-max_seq_len:]
    input_ids = torch.tensor([window], dtype=torch.long, device=device)
    logits, _ = model(input_ids)
    next_token = int(torch.argmax(logits[0, -1]).detach().cpu())
    tokens.append(next_token)
  return tokens
