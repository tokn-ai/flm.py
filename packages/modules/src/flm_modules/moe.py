"""Mixture-of-experts layers."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from flm_modules.feed_forward import SwiGLU


class DeepSeekTopKRouter(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_routed_experts: int,
  ) -> None:
    super().__init__()
    self.d_model = d_model
    self.n_routed_experts = n_routed_experts
    self.weight = nn.Parameter(torch.empty(n_routed_experts, d_model))
    self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.reshape(-1, self.d_model)
    return F.linear(
      hidden_states.to(torch.float32),
      self.weight.to(torch.float32),
    )


class DeepSeekMoE(nn.Module):
  """DeepSeek-style MoE layer with routed and shared SwiGLU experts."""

  def __init__(
    self,
    d_model: int,
    d_ff: int,
    n_routed_experts: int,
    n_shared_experts: int,
    n_experts_per_token: int,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    bias: bool = False,
  ) -> None:
    super().__init__()
    if n_routed_experts <= 0:
      raise ValueError("n_routed_experts must be positive")
    if n_shared_experts < 0:
      raise ValueError("n_shared_experts must be non-negative")
    if n_experts_per_token <= 0 or n_experts_per_token > n_routed_experts:
      raise ValueError("n_experts_per_token must be in [1, n_routed_experts]")
    if n_group <= 0 or n_routed_experts % n_group != 0:
      raise ValueError("n_group must divide n_routed_experts")
    if topk_group <= 0 or topk_group > n_group:
      raise ValueError("topk_group must be in [1, n_group]")
    group_size = n_routed_experts // n_group
    if group_size < 2:
      raise ValueError("each expert group must contain at least two experts")
    if n_experts_per_token > topk_group * group_size:
      raise ValueError("n_experts_per_token exceeds selected expert groups")

    self.d_model = d_model
    self.d_ff = d_ff
    self.n_routed_experts = n_routed_experts
    self.n_shared_experts = n_shared_experts
    self.n_experts_per_token = n_experts_per_token
    self.n_group = n_group
    self.topk_group = topk_group
    self.norm_topk_prob = norm_topk_prob
    self.routed_scaling_factor = routed_scaling_factor

    self.gate = DeepSeekTopKRouter(d_model, n_routed_experts)
    self.experts = nn.ModuleList(
      [SwiGLU(d_model=d_model, d_ff=d_ff, bias=bias) for _ in range(n_routed_experts)]
    )
    self.shared_experts = (
      SwiGLU(d_model=d_model, d_ff=d_ff * n_shared_experts, bias=bias)
      if n_shared_experts > 0
      else None
    )

  def route_tokens_to_experts(
    self,
    router_logits: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    router_probs = router_logits.sigmoid()
    router_scores = router_probs + self.gate.e_score_correction_bias
    group_size = self.n_routed_experts // self.n_group
    group_scores = (
      router_scores.view(-1, self.n_group, group_size).topk(2, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(
      group_scores,
      k=self.topk_group,
      dim=-1,
      sorted=False,
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
      group_mask.unsqueeze(-1)
      .expand(-1, self.n_group, group_size)
      .reshape(-1, self.n_routed_experts)
    )
    scores_for_choice = router_scores.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(
      scores_for_choice,
      k=self.n_experts_per_token,
      dim=-1,
      sorted=False,
    )[1]
    topk_weights = router_probs.gather(1, topk_indices)
    if self.norm_topk_prob:
      topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * self.routed_scaling_factor
    return topk_indices, topk_weights

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    original_shape = hidden_states.shape
    flat_hidden_states = hidden_states.reshape(-1, self.d_model)
    router_logits = self.gate(hidden_states)
    topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
    routed = self._route_to_experts(
      flat_hidden_states,
      topk_indices,
      topk_weights,
    ).view(original_shape)

    if self.shared_experts is None:
      return routed
    return routed + self.shared_experts(hidden_states)

  def _route_to_experts(
    self,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
  ) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(
      topk_indices,
      num_classes=self.n_routed_experts,
    ).permute(2, 1, 0)

    for expert_idx in torch.nonzero(expert_mask.sum(dim=(-1, -2)), as_tuple=False):
      expert_idx = expert_idx.item()
      topk_pos, token_idx = torch.where(expert_mask[expert_idx])
      current_hidden_states = self.experts[expert_idx](hidden_states[token_idx])
      current_hidden_states = current_hidden_states * topk_weights[
        token_idx,
        topk_pos,
        None,
      ].to(current_hidden_states.dtype)
      final_hidden_states.index_add_(
        0,
        token_idx,
        current_hidden_states.to(final_hidden_states.dtype),
      )

    return final_hidden_states


class DeepSeekV4TopKRouter(nn.Module):
  """DeepSeek V4 learned top-k router."""

  def __init__(
    self,
    d_model: int,
    n_routed_experts: int,
    n_experts_per_token: int,
    routed_scaling_factor: float = 1.5,
  ) -> None:
    super().__init__()
    if n_routed_experts <= 0:
      raise ValueError("n_routed_experts must be positive")
    if n_experts_per_token <= 0 or n_experts_per_token > n_routed_experts:
      raise ValueError("n_experts_per_token must be in [1, n_routed_experts]")
    self.d_model = d_model
    self.n_routed_experts = n_routed_experts
    self.n_experts_per_token = n_experts_per_token
    self.routed_scaling_factor = routed_scaling_factor
    self.weight = nn.Parameter(torch.empty(n_routed_experts, d_model))
    self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def forward(
    self,
    hidden_states: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.d_model)
    logits = F.linear(flat, self.weight)
    scores = F.softplus(logits).sqrt()
    indices = torch.topk(
      scores + self.e_score_correction_bias,
      self.n_experts_per_token,
      dim=-1,
      sorted=False,
    ).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices
