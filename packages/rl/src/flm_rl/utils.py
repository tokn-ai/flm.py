"""Shared helpers for language-model RL losses."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


class CausalLM(Protocol):
  def __call__(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


def model_logits(model: CausalLM | nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
  output = model(input_ids)
  if isinstance(output, tuple):
    logits = output[0]
  else:
    logits = output
  if not isinstance(logits, torch.Tensor):
    raise TypeError("model must return logits or (logits, loss)")
  return logits


def sequence_log_probs(
  model: CausalLM | nn.Module,
  input_ids: torch.Tensor,
) -> torch.Tensor:
  """Return next-token log-probabilities for each token after the prompt token."""

  logits = model_logits(model, input_ids[:, :-1])
  labels = input_ids[:, 1:]
  if logits.shape[:2] != labels.shape:
    raise ValueError("model logits must align with next-token labels")
  log_probs = F.log_softmax(logits, dim=-1)
  return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def sequence_log_probs_and_entropy(
  model: CausalLM | nn.Module,
  input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  logits = model_logits(model, input_ids[:, :-1])
  labels = input_ids[:, 1:]
  if logits.shape[:2] != labels.shape:
    raise ValueError("model logits must align with next-token labels")
  log_probs = F.log_softmax(logits, dim=-1)
  probs = log_probs.exp()
  token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
  entropy = -(probs * log_probs).sum(dim=-1)
  return token_log_probs, entropy


def masked_mean(
  values: torch.Tensor,
  mask: torch.Tensor | None = None,
  eps: float = 1e-8,
) -> torch.Tensor:
  if mask is None:
    return values.mean()
  mask = mask.to(dtype=values.dtype, device=values.device)
  return (values * mask).sum() / mask.sum().clamp_min(eps)


def normalize_masked(
  values: torch.Tensor,
  mask: torch.Tensor | None = None,
  eps: float = 1e-8,
) -> torch.Tensor:
  mean = masked_mean(values, mask, eps=eps)
  if mask is None:
    var = (values - mean).pow(2).mean()
    return (values - mean) / torch.sqrt(var + eps)
  mask = mask.to(dtype=values.dtype, device=values.device)
  var = ((values - mean).pow(2) * mask).sum() / mask.sum().clamp_min(eps)
  return (values - mean) / torch.sqrt(var + eps)


def trainable_parameters(*modules: object) -> list[nn.Parameter]:
  params: list[nn.Parameter] = []
  for module in modules:
    if isinstance(module, nn.Module):
      params.extend(param for param in module.parameters() if param.requires_grad)
  return params


def compute_group_advantages(
  rewards: torch.Tensor,
  group_ids: torch.Tensor,
  eps: float = 1e-8,
) -> torch.Tensor:
  """Normalize scalar rewards within each prompt group."""

  if rewards.ndim != 1:
    raise ValueError("rewards must have shape (batch,)")
  if group_ids.shape != rewards.shape:
    raise ValueError("group_ids must have the same shape as rewards")

  advantages = torch.empty_like(rewards)
  for group_id in group_ids.unique(sorted=False):
    group_mask = group_ids == group_id
    group_rewards = rewards[group_mask]
    centered = group_rewards - group_rewards.mean()
    if group_rewards.numel() > 1:
      scale = group_rewards.std(unbiased=False).clamp_min(eps)
      advantages[group_mask] = centered / scale
    else:
      advantages[group_mask] = centered
  return advantages
