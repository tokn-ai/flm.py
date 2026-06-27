"""Proximal Policy Optimization for causal language models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flm_rl.utils import (
  CausalLM,
  masked_mean,
  normalize_masked,
  sequence_log_probs,
  sequence_log_probs_and_entropy,
  trainable_parameters,
)


@dataclass(frozen=True)
class PPOConfig:
  clip_range: float = 0.2
  value_clip_range: float = 0.2
  value_coef: float = 0.5
  entropy_coef: float = 0.0
  kl_coef: float = 0.0
  max_grad_norm: float | None = 1.0
  normalize_advantages: bool = True


@dataclass(frozen=True)
class PPOBatch:
  input_ids: torch.Tensor
  old_log_probs: torch.Tensor
  advantages: torch.Tensor
  action_mask: torch.Tensor | None = None
  returns: torch.Tensor | None = None
  old_values: torch.Tensor | None = None


@dataclass(frozen=True)
class PPOMetrics:
  loss: float
  policy_loss: float
  value_loss: float
  entropy: float
  approx_kl: float
  clip_fraction: float
  ref_kl: float


class PPOTrainer:
  def __init__(
    self,
    policy_model: CausalLM | nn.Module,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig | None = None,
    value_model: nn.Module | None = None,
    reference_model: CausalLM | nn.Module | None = None,
  ) -> None:
    self.policy_model = policy_model
    self.optimizer = optimizer
    self.config = config or PPOConfig()
    self.value_model = value_model
    self.reference_model = reference_model

  def step(self, batch: PPOBatch) -> PPOMetrics:
    self.optimizer.zero_grad(set_to_none=True)
    log_probs, entropy = sequence_log_probs_and_entropy(
      self.policy_model,
      batch.input_ids,
    )
    _validate_action_shape(log_probs, batch.old_log_probs, "old_log_probs")
    advantages = batch.advantages.to(device=log_probs.device, dtype=log_probs.dtype)
    _validate_action_shape(log_probs, advantages, "advantages")

    action_mask = _action_mask(batch.action_mask, log_probs)
    if self.config.normalize_advantages:
      advantages = normalize_masked(advantages, action_mask)

    old_log_probs = batch.old_log_probs.to(
      device=log_probs.device,
      dtype=log_probs.dtype,
    )
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(
      1.0 - self.config.clip_range,
      1.0 + self.config.clip_range,
    )
    policy_loss = -masked_mean(
      torch.minimum(ratio * advantages, clipped_ratio * advantages),
      action_mask,
    )
    clip_fraction = masked_mean(
      ((ratio - 1.0).abs() > self.config.clip_range).to(log_probs.dtype),
      action_mask,
    )
    approx_kl = masked_mean((ratio - 1.0) - log_ratio, action_mask)
    entropy_loss = masked_mean(entropy, action_mask)

    value_loss = log_probs.new_tensor(0.0)
    if self.value_model is not None or batch.returns is not None:
      value_loss = self._value_loss(batch, log_probs)

    ref_kl = log_probs.new_tensor(0.0)
    if self.reference_model is not None and self.config.kl_coef != 0.0:
      with torch.no_grad():
        ref_log_probs = sequence_log_probs(self.reference_model, batch.input_ids)
      ref_kl = masked_mean(log_probs - ref_log_probs.to(log_probs.device), action_mask)

    loss = (
      policy_loss
      + self.config.value_coef * value_loss
      - self.config.entropy_coef * entropy_loss
      + self.config.kl_coef * ref_kl
    )
    loss.backward()
    if self.config.max_grad_norm is not None:
      torch.nn.utils.clip_grad_norm_(
        trainable_parameters(self.policy_model, self.value_model),
        self.config.max_grad_norm,
      )
    self.optimizer.step()

    return PPOMetrics(
      loss=float(loss.detach().cpu()),
      policy_loss=float(policy_loss.detach().cpu()),
      value_loss=float(value_loss.detach().cpu()),
      entropy=float(entropy_loss.detach().cpu()),
      approx_kl=float(approx_kl.detach().cpu()),
      clip_fraction=float(clip_fraction.detach().cpu()),
      ref_kl=float(ref_kl.detach().cpu()),
    )

  def _value_loss(self, batch: PPOBatch, like: torch.Tensor) -> torch.Tensor:
    if self.value_model is None:
      raise ValueError("returns require a value_model")
    if batch.returns is None:
      raise ValueError("value_model requires returns")

    values = self.value_model(batch.input_ids[:, :-1])
    if isinstance(values, tuple):
      values = values[0]
    if values.ndim == 3 and values.shape[-1] == 1:
      values = values.squeeze(-1)
    if not isinstance(values, torch.Tensor):
      raise TypeError("value_model must return values")
    _validate_action_shape(like, values, "values")

    action_mask = _action_mask(batch.action_mask, like)
    returns = batch.returns.to(device=like.device, dtype=like.dtype)
    _validate_action_shape(like, returns, "returns")
    values = values.to(device=like.device, dtype=like.dtype)
    value_loss = (values - returns).pow(2)
    if batch.old_values is not None:
      old_values = batch.old_values.to(device=like.device, dtype=like.dtype)
      _validate_action_shape(like, old_values, "old_values")
      clipped = old_values + (values - old_values).clamp(
        -self.config.value_clip_range,
        self.config.value_clip_range,
      )
      value_loss = torch.maximum(value_loss, (clipped - returns).pow(2))
    return 0.5 * masked_mean(value_loss, action_mask)


def _action_mask(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | None:
  if mask is None:
    return None
  _validate_action_shape(like, mask, "action_mask")
  return mask.to(device=like.device, dtype=like.dtype)


def _validate_action_shape(
  expected: torch.Tensor,
  actual: torch.Tensor,
  name: str,
) -> None:
  if actual.shape != expected.shape:
    raise ValueError(f"{name} must have shape {tuple(expected.shape)}")
