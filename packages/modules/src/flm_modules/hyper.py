"""DeepSeek V4 hyper-connection layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class UnweightedRMSNorm(nn.Module):
  def __init__(self, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
    return x * scale.to(x.dtype)


class DeepSeekV4HyperConnection(nn.Module):
  """Manifold-constrained hyper-connection for DeepSeek V4 residual streams."""

  def __init__(
    self,
    d_model: int,
    hc_mult: int,
    hc_sinkhorn_iters: int = 3,
    hc_eps: float = 1e-6,
    rms_norm_eps: float = 1e-6,
    initializer_range: float = 0.02,
  ) -> None:
    super().__init__()
    if d_model <= 0:
      raise ValueError("d_model must be positive")
    if hc_mult <= 0:
      raise ValueError("hc_mult must be positive")
    if hc_sinkhorn_iters <= 0:
      raise ValueError("hc_sinkhorn_iters must be positive")

    self.hc_mult = hc_mult
    self.hc_sinkhorn_iters = hc_sinkhorn_iters
    self.hc_eps = hc_eps
    self.input_norm = UnweightedRMSNorm(eps=rms_norm_eps)
    mix = (2 + hc_mult) * hc_mult
    self.fn = nn.Parameter(torch.empty(mix, hc_mult * d_model))
    self.base = nn.Parameter(torch.empty(mix))
    self.scale = nn.Parameter(torch.empty(3))
    self.reset_parameters(initializer_range)

  def reset_parameters(self, initializer_range: float = 0.02) -> None:
    nn.init.normal_(self.fn, mean=0.0, std=initializer_range)
    nn.init.zeros_(self.base)
    nn.init.ones_(self.scale)

  def forward(
    self,
    hidden_streams: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc = self.hc_mult
    if hidden_streams.shape[-2] != hc:
      raise ValueError("input stream dimension must match hc_mult")

    flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
    pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(
      [hc, hc, hc * hc],
      dim=-1,
    )
    pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = self.scale.unbind(0)

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(
      hc,
      hc,
    )
    comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    for _ in range(self.hc_sinkhorn_iters - 1):
      comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
      comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
    return post, comb, collapsed


class DeepSeekV4HyperHead(nn.Module):
  """Final DeepSeek V4 hyper-connection stream collapse."""

  def __init__(
    self,
    d_model: int,
    hc_mult: int,
    hc_eps: float = 1e-6,
    rms_norm_eps: float = 1e-6,
    initializer_range: float = 0.02,
  ) -> None:
    super().__init__()
    if d_model <= 0:
      raise ValueError("d_model must be positive")
    if hc_mult <= 0:
      raise ValueError("hc_mult must be positive")

    self.hc_mult = hc_mult
    self.input_norm = UnweightedRMSNorm(eps=rms_norm_eps)
    self.eps = hc_eps
    self.hc_fn = nn.Parameter(torch.empty(hc_mult, hc_mult * d_model))
    self.hc_base = nn.Parameter(torch.empty(hc_mult))
    self.hc_scale = nn.Parameter(torch.empty(1))
    self.reset_parameters(initializer_range)

  def reset_parameters(self, initializer_range: float = 0.02) -> None:
    nn.init.normal_(self.hc_fn, mean=0.0, std=initializer_range)
    nn.init.zeros_(self.hc_base)
    nn.init.ones_(self.hc_scale)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] != self.hc_mult:
      raise ValueError("input stream dimension must match hc_mult")

    flat = self.input_norm(x.flatten(start_dim=2).float())
    mixes = F.linear(flat, self.hc_fn.float())
    pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
    return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)
