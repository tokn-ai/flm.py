"""Group Relative Policy Optimization for causal language models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flm_rl.utils import (
  CausalLM,
  compute_group_advantages,
  masked_mean,
  sequence_log_probs,
  trainable_parameters,
)


@dataclass(frozen=True)
class GRPOConfig:
  clip_range: float = 0.2
  kl_coef: float = 0.0
  max_grad_norm: float | None = 1.0
  group_eps: float = 1e-8


@dataclass(frozen=True)
class GRPOBatch:
  input_ids: torch.Tensor
  old_log_probs: torch.Tensor
  rewards: torch.Tensor
  group_ids: torch.Tensor
  action_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class GRPOMetrics:
  loss: float
  policy_loss: float
  approx_kl: float
  clip_fraction: float
  ref_kl: float
  reward_mean: float


class GRPOTrainer:
  def __init__(
    self,
    policy_model: CausalLM | nn.Module,
    optimizer: torch.optim.Optimizer,
    config: GRPOConfig | None = None,
    reference_model: CausalLM | nn.Module | None = None,
  ) -> None:
    self.policy_model = policy_model
    self.optimizer = optimizer
    self.config = config or GRPOConfig()
    self.reference_model = reference_model

  def step(self, batch: GRPOBatch) -> GRPOMetrics:
    self.optimizer.zero_grad(set_to_none=True)
    log_probs = sequence_log_probs(self.policy_model, batch.input_ids)
    _validate_action_shape(log_probs, batch.old_log_probs, "old_log_probs")
    action_mask = _action_mask(batch.action_mask, log_probs)

    advantages = compute_group_advantages(
      batch.rewards.to(device=log_probs.device, dtype=log_probs.dtype),
      batch.group_ids.to(device=log_probs.device),
      eps=self.config.group_eps,
    )
    token_advantages = advantages[:, None].expand_as(log_probs)

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
      torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages),
      action_mask,
    )
    approx_kl = masked_mean((ratio - 1.0) - log_ratio, action_mask)
    clip_fraction = masked_mean(
      ((ratio - 1.0).abs() > self.config.clip_range).to(log_probs.dtype),
      action_mask,
    )

    ref_kl = log_probs.new_tensor(0.0)
    if self.reference_model is not None and self.config.kl_coef != 0.0:
      with torch.no_grad():
        ref_log_probs = sequence_log_probs(self.reference_model, batch.input_ids)
      ref_kl = masked_mean(log_probs - ref_log_probs.to(log_probs.device), action_mask)

    loss = policy_loss + self.config.kl_coef * ref_kl
    loss.backward()
    if self.config.max_grad_norm is not None:
      torch.nn.utils.clip_grad_norm_(
        trainable_parameters(self.policy_model),
        self.config.max_grad_norm,
      )
    self.optimizer.step()

    return GRPOMetrics(
      loss=float(loss.detach().cpu()),
      policy_loss=float(policy_loss.detach().cpu()),
      approx_kl=float(approx_kl.detach().cpu()),
      clip_fraction=float(clip_fraction.detach().cpu()),
      ref_kl=float(ref_kl.detach().cpu()),
      reward_mean=float(batch.rewards.float().mean().detach().cpu()),
    )


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
