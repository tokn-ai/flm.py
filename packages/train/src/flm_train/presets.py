"""Preset training workflows used by experiments and smoke tests."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from flm_datasets import get_tokenizer
from flm_modules import configure_adamw, configure_muon, configure_normuon

from flm_train.data import build_training_dataset
from flm_train.models import build_model
from flm_train.schedules import OptimizerSchedule
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
  on_batch_size_resolved: Callable[[int], None] | None = None,
  checkpoint_dir: Path | None = None,
  on_checkpoint: Callable[[Path, int], None] | None = None,
) -> TrainingResult:
  torch.manual_seed(config.loop.seed)

  encoding = get_tokenizer(config.data.encoding_name)
  config = _config_with_resolved_batch_size(
    config,
    vocab_size=encoding.n_vocab,
    on_batch_size_resolved=on_batch_size_resolved,
  )
  model = build_model(
    config,
    vocab_size=encoding.n_vocab,
  ).to(device=config.loop.device, dtype=_torch_dtype(config.loop.dtype))
  dataset_bundle = build_training_dataset(config)
  eval_bundle = None
  if config.eval is not None:
    eval_config = replace(
      config,
      data=replace(config.data, split=config.eval.split),
    )
    eval_bundle = build_training_dataset(eval_config)
  optimizer = _build_optimizer(config, model)
  schedule = (
    None
    if config.loop.steps == 0
    else OptimizerSchedule(
      optimizer,
      total_steps=config.loop.steps,
      config=config.schedule,
    )
  )
  trainer = LanguageModelTrainer(
    model=model,
    optimizer=optimizer,
    optimizer_schedule=schedule,
    dataloader=dataset_bundle.dataloader,
    device=config.loop.device,
    steps=config.loop.steps,
    bytes_per_token=_bytes_per_token(
      dataset_bundle.byte_count,
      dataset_bundle.token_count,
    ),
    max_grad_norm=config.optimizer.max_grad_norm,
    on_step=on_step,
    eval_every_steps=config.eval.every_steps if config.eval is not None else None,
    evaluate=None
    if config.eval is None or eval_bundle is None
    else lambda step, model: evaluate_language_model(
      model=model,
      dataloader=eval_bundle.dataloader,
      device=config.loop.device,
      split=config.eval.split,
      bytes_per_token=_bytes_per_token(
        eval_bundle.byte_count,
        eval_bundle.token_count,
      ),
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
    checkpoint=config.checkpoint,
    checkpoint_dir=checkpoint_dir,
    on_checkpoint=on_checkpoint,
  )
  step_metrics = trainer.train()

  return TrainingResult(
    losses=[metrics.loss for metrics in step_metrics],
    token_count=dataset_bundle.token_count,
    file_count=dataset_bundle.file_count,
    byte_count=dataset_bundle.byte_count,
  )


def _config_with_resolved_batch_size(
  config: TrainConfig,
  *,
  vocab_size: int,
  on_batch_size_resolved: Callable[[int], None] | None = None,
) -> TrainConfig:
  if config.loop.batch_size != "auto":
    return config
  batch_size = probe_auto_batch_size(
    config=config,
    vocab_size=vocab_size,
  )
  if on_batch_size_resolved is not None:
    on_batch_size_resolved(batch_size)
  return replace(config, loop=replace(config.loop, batch_size=batch_size))


def probe_auto_batch_size(
  *,
  config: TrainConfig,
  vocab_size: int,
) -> int:
  model = build_model(
    config,
    vocab_size=vocab_size,
  ).to(device=config.loop.device, dtype=_torch_dtype(config.loop.dtype))
  optimizer = _build_optimizer(config, model)
  try:
    return auto_batch_size(
      model=model,
      optimizer=optimizer,
      vocab_size=vocab_size,
      seq_len=config.data.seq_len,
      device=config.loop.device,
      target_fraction=config.loop.batch_size_vram_fraction,
    )
  finally:
    del optimizer, model
    if config.loop.device.startswith("cuda") and torch.cuda.is_available():
      torch.cuda.empty_cache()


def auto_batch_size(
  *,
  model: LanguageModel,
  optimizer: torch.optim.Optimizer,
  vocab_size: int,
  seq_len: int,
  device: str,
  target_fraction: float,
) -> int:
  if not device.startswith("cuda"):
    raise ValueError("loop.batch_size='auto' requires a CUDA device")
  if not torch.cuda.is_available():
    raise RuntimeError("loop.batch_size='auto' requires CUDA to be available")
  if not 0 < target_fraction <= 1:
    raise ValueError("loop.batch_size_vram_fraction must be in (0, 1]")

  cuda_device = torch.device(device)
  _, total_bytes = torch.cuda.mem_get_info(cuda_device)
  target_bytes = int(total_bytes * target_fraction)

  if not _batch_size_fits(
    model=model,
    optimizer=optimizer,
    batch_size=1,
    seq_len=seq_len,
    vocab_size=vocab_size,
    device=cuda_device,
    target_bytes=target_bytes,
  ):
    raise RuntimeError(
      "batch_size=1 does not fit within "
      f"{target_fraction:.0%} of CUDA VRAM for seq_len={seq_len}"
    )

  low = 1
  high = 2
  while _batch_size_fits(
    model=model,
    optimizer=optimizer,
    batch_size=high,
    seq_len=seq_len,
    vocab_size=vocab_size,
    device=cuda_device,
    target_bytes=target_bytes,
  ):
    low = high
    high *= 2

  while high - low > 1:
    candidate = (low + high) // 2
    if _batch_size_fits(
      model=model,
      optimizer=optimizer,
      batch_size=candidate,
      seq_len=seq_len,
      vocab_size=vocab_size,
      device=cuda_device,
      target_bytes=target_bytes,
    ):
      low = candidate
    else:
      high = candidate
  return low


def _batch_size_fits(
  *,
  model: LanguageModel,
  optimizer: torch.optim.Optimizer,
  batch_size: int,
  seq_len: int,
  vocab_size: int,
  device: torch.device,
  target_bytes: int,
) -> bool:
  was_training = model.training
  model.train()
  input_ids = None
  targets = None
  loss = None
  try:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    input_ids = torch.randint(
      low=0,
      high=vocab_size,
      size=(batch_size, seq_len),
      dtype=torch.long,
      device=device,
    )
    targets = torch.randint(
      low=0,
      high=vocab_size,
      size=(batch_size, seq_len),
      dtype=torch.long,
      device=device,
    )
    _, loss = model(input_ids, targets, return_logits=False)
    if loss is None:
      raise RuntimeError("training loss was not produced during batch-size probe")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes <= target_bytes
  except torch.cuda.OutOfMemoryError:
    return False
  except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
      return False
    raise
  finally:
    del input_ids, targets, loss
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    model.train(was_training)
    torch.cuda.empty_cache()


def _build_optimizer(
  config: TrainConfig,
  model: torch.nn.Module,
) -> torch.optim.Optimizer:
  if config.optimizer.kind == "adamw":
    return configure_adamw(
      model,
      learning_rate=config.optimizer.learning_rate,
      weight_decay=config.optimizer.weight_decay,
    )
  if config.optimizer.kind == "muon":
    return configure_muon(
      model,
      learning_rate=config.optimizer.learning_rate,
      weight_decay=config.optimizer.weight_decay,
    )
  if config.optimizer.kind == "normuon":
    return configure_normuon(
      model,
      learning_rate=config.optimizer.learning_rate,
      weight_decay=config.optimizer.weight_decay,
    )
  raise ValueError(f"unsupported optimizer.kind: {config.optimizer.kind}")


def evaluate_language_model(
  *,
  model: LanguageModel,
  dataloader,
  device: str,
  split: str,
  bytes_per_token: float,
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
      _, loss = model(input_ids, targets, return_logits=False)
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
    bits_per_byte=_loss_to_bits_per_byte(loss=loss, bytes_per_token=bytes_per_token),
    tokens=total_tokens,
  )


def _torch_dtype(value: str) -> torch.dtype:
  if value == "float32":
    return torch.float32
  if value == "float16":
    return torch.float16
  if value == "bfloat16":
    return torch.bfloat16
  raise ValueError(f"unsupported torch dtype: {value}")


def _bytes_per_token(byte_count: int, token_count: int) -> float:
  if token_count <= 0:
    return 0.0
  return byte_count / token_count


def _loss_to_bits_per_byte(*, loss: float, bytes_per_token: float) -> float:
  if bytes_per_token <= 0:
    return 0.0
  return loss / math.log(2.0) / bytes_per_token


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
      prompt_log_probs = prompt_token_log_probs(
        model=model,
        prompt_tokens=prompt_tokens,
        device=device,
        max_seq_len=max_seq_len,
      )
      generated_tokens = greedy_decode(
        model=model,
        prompt_tokens=prompt_tokens,
        encoding=encoding,
        device=device,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
      )
      output_tokens = prompt_tokens + generated_tokens.tokens
      text = encoding.decode(output_tokens)
      samples.append(
        RolloutSample(
          name=prompt.name,
          prompt=prompt.prompt,
          prompt_tokens=tuple(prompt_tokens),
          prompt_log_probs=tuple(prompt_log_probs),
          tokens=tuple(generated_tokens.tokens),
          token_texts=tuple(generated_tokens.token_texts),
          log_probs=tuple(generated_tokens.log_probs),
          entropy=tuple(generated_tokens.entropy),
          top_tokens=tuple(tuple(tokens) for tokens in generated_tokens.top_tokens),
          top_token_texts=tuple(
            tuple(texts) for texts in generated_tokens.top_token_texts
          ),
          top_log_probs=tuple(
            tuple(log_probs) for log_probs in generated_tokens.top_log_probs
          ),
          text=text,
        )
      )
  return RolloutBatch(step=step, samples=tuple(samples))


def prompt_token_log_probs(
  *,
  model: LanguageModel,
  prompt_tokens: list[int],
  device: str,
  max_seq_len: int,
) -> list[float]:
  values: list[float] = []
  for index in range(1, len(prompt_tokens)):
    window = prompt_tokens[max(0, index - max_seq_len) : index]
    input_ids = torch.tensor([window], dtype=torch.long, device=device)
    logits, _ = model(input_ids)
    log_probs = torch.log_softmax(logits[0, -1], dim=-1)
    values.append(float(log_probs[prompt_tokens[index]].detach().cpu()))
  return values


def greedy_decode(
  *,
  model: LanguageModel,
  prompt_tokens: list[int],
  encoding,
  device: str,
  max_seq_len: int,
  max_new_tokens: int,
) -> RolloutGeneration:
  if not prompt_tokens:
    raise ValueError("rollout prompt must not be empty")
  tokens = list(prompt_tokens)
  generated_tokens: list[int] = []
  token_texts: list[str] = []
  log_prob_values: list[float] = []
  entropy_values: list[float] = []
  top_token_values: list[list[int]] = []
  top_token_texts: list[list[str]] = []
  top_log_prob_values: list[list[float]] = []
  for _ in range(max_new_tokens):
    window = tokens[-max_seq_len:]
    input_ids = torch.tensor([window], dtype=torch.long, device=device)
    logits, _ = model(input_ids)
    log_probs = torch.log_softmax(logits[0, -1], dim=-1)
    probs = torch.exp(log_probs)
    top_log_probs, top_tokens = torch.topk(log_probs, k=min(10, log_probs.numel()))
    next_token = int(top_tokens[0].detach().cpu())
    generated_tokens.append(next_token)
    token_texts.append(encoding.decode([next_token]))
    log_prob_values.append(float(top_log_probs[0].detach().cpu()))
    entropy_values.append(float(-(probs * log_probs).sum().detach().cpu()))
    top_token_values.append([int(token.detach().cpu()) for token in top_tokens])
    top_token_texts.append(
      [encoding.decode([int(token.detach().cpu())]) for token in top_tokens]
    )
    top_log_prob_values.append(
      [float(log_prob.detach().cpu()) for log_prob in top_log_probs]
    )
    tokens.append(next_token)
  return RolloutGeneration(
    tokens=generated_tokens,
    token_texts=token_texts,
    log_probs=log_prob_values,
    entropy=entropy_values,
    top_tokens=top_token_values,
    top_token_texts=top_token_texts,
    top_log_probs=top_log_prob_values,
  )


@dataclass(frozen=True)
class RolloutGeneration:
  tokens: list[int]
  token_texts: list[str]
  log_probs: list[float]
  entropy: list[float]
  top_tokens: list[list[int]]
  top_token_texts: list[list[str]]
  top_log_probs: list[list[float]]
