"""Muon optimizer helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from flm_modules.optim.adamw import CautiousAdamW
from flm_modules.optim.composite import CompositeOptimizer


class Muon(Optimizer):
  """Muon optimizer for 2D matrix parameters."""

  def __init__(
    self,
    params: Iterable[dict[str, object]],
    *,
    lr: float = 3e-4,
    momentum: float = 0.95,
    weight_decay: float = 0.1,
    nesterov: bool = True,
    ns_steps: int = 5,
  ) -> None:
    if lr < 0.0:
      raise ValueError(f"invalid learning rate: {lr}")
    if momentum < 0.0:
      raise ValueError(f"invalid momentum value: {momentum}")
    if weight_decay < 0.0:
      raise ValueError(f"invalid weight_decay value: {weight_decay}")
    if ns_steps < 1:
      raise ValueError(f"invalid ns_steps value: {ns_steps}")

    defaults = {
      "lr": lr,
      "momentum": momentum,
      "weight_decay": weight_decay,
      "nesterov": nesterov,
      "ns_steps": ns_steps,
    }
    super().__init__(params, defaults)
    for group in self.param_groups:
      for param in group["params"]:
        if param.ndim != 2:
          raise ValueError(
            "Muon only supports 2D parameters, "
            f"but found parameter with shape {tuple(param.shape)}"
          )

  @torch.no_grad()
  def step(self, closure: Callable[[], object] | None = None) -> object | None:
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      self._muon_step(group)
    return loss

  def _muon_step(self, group: dict[str, object]) -> None:
    lr = float(group["lr"])
    momentum = float(group["momentum"])
    weight_decay = float(group["weight_decay"])
    nesterov = bool(group["nesterov"])
    ns_steps = int(group["ns_steps"])

    for param in group["params"]:
      if param.grad is None:
        continue
      grad = param.grad
      if grad.is_sparse:
        raise RuntimeError("Muon does not support sparse gradients")
      if weight_decay:
        param.mul_(1.0 - lr * weight_decay)

      state = self.state[param]
      if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(param)
      buffer = state["momentum_buffer"]
      buffer.lerp_(grad, 1.0 - momentum)
      update = grad.lerp(buffer, momentum) if nesterov else buffer
      update = _orthogonalize_update(update, ns_steps=ns_steps)
      param.add_(update, alpha=-_adjust_muon_lr(lr, param.shape))


class NorMuon(Optimizer):
  """Portable NorMuon with Polar Express and low-rank variance reduction."""

  def __init__(
    self,
    params: Iterable[nn.Parameter],
    *,
    lr: float = 3e-4,
    momentum: float = 0.95,
    beta2: float = 0.95,
    weight_decay: float = 0.1,
  ) -> None:
    if not 0 <= momentum < 1:
      raise ValueError(f"invalid momentum value: {momentum}")
    if not 0 <= beta2 < 1:
      raise ValueError(f"invalid beta2 value: {beta2}")
    defaults = {
      "lr": lr,
      "momentum": momentum,
      "beta2": beta2,
      "weight_decay": weight_decay,
    }
    super().__init__(params, defaults)
    for group in self.param_groups:
      for param in group["params"]:
        if param.ndim != 2:
          raise ValueError("NorMuon only supports 2D matrix parameters")

  @torch.no_grad()
  def step(self, closure: Callable[[], object] | None = None) -> object | None:
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      for param in group["params"]:
        if param.grad is None:
          continue
        if param.grad.is_sparse:
          raise RuntimeError("NorMuon does not support sparse gradients")
        self._step_parameter(param, group)
    return loss

  def _step_parameter(
    self,
    param: nn.Parameter,
    group: dict[str, object],
  ) -> None:
    grad = param.grad.float()
    momentum = float(group["momentum"])
    beta2 = float(group["beta2"])
    lr = _adjust_muon_lr(float(group["lr"]), param.shape)
    weight_decay = float(group["weight_decay"])
    state = self.state[param]

    momentum_buffer = state.setdefault("momentum_buffer", torch.zeros_like(grad))
    momentum_buffer.lerp_(grad, 1.0 - momentum)
    update = grad.lerp(momentum_buffer, momentum)
    update = _polar_express(update)

    red_dim = -1 if update.shape[-2] >= update.shape[-1] else -2
    reduced_shape = list(update.shape)
    reduced_shape[red_dim] = 1
    second_momentum = state.setdefault(
      "second_momentum_buffer",
      torch.zeros(reduced_shape, dtype=torch.float32, device=param.device),
    )
    update = _normuon_variance_reduction(
      update,
      second_momentum,
      beta2=beta2,
      red_dim=red_dim,
    )

    if weight_decay:
      cautious_mask = (update * param.float()) >= 0
      param.add_(param * cautious_mask, alpha=-(lr * weight_decay))
    param.add_(update.to(param.dtype), alpha=-lr)


def configure_muon(
  model: nn.Module,
  learning_rate: float = 3e-4,
  weight_decay: float = 0.1,
  momentum: float = 0.95,
  nesterov: bool = True,
  ns_steps: int = 5,
  adamw_betas: tuple[float, float] = (0.9, 0.95),
  adamw_eps: float = 1e-8,
) -> CompositeOptimizer:
  muon_params: list[nn.Parameter] = []
  adamw_params: list[nn.Parameter] = []

  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if _use_matrix_optimizer(name, param):
      muon_params.append(param)
    else:
      adamw_params.append(param)

  optimizers: list[torch.optim.Optimizer] = []
  if muon_params:
    optimizers.append(
      Muon(
        muon_params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        ns_steps=ns_steps,
      )
    )
  if adamw_params:
    optimizers.append(
      CautiousAdamW(
        [{"params": adamw_params, "weight_decay": weight_decay}],
        lr=learning_rate,
        betas=adamw_betas,
        eps=adamw_eps,
      )
    )
  return CompositeOptimizer(optimizers)


def configure_normuon(
  model: nn.Module,
  learning_rate: float = 3e-4,
  weight_decay: float = 0.1,
  momentum: float = 0.95,
  beta2: float = 0.95,
  adamw_betas: tuple[float, float] = (0.9, 0.95),
  adamw_eps: float = 1e-8,
) -> CompositeOptimizer:
  matrix_params: list[nn.Parameter] = []
  adamw_params: list[nn.Parameter] = []
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    (matrix_params if _use_matrix_optimizer(name, param) else adamw_params).append(
      param
    )

  optimizers: list[torch.optim.Optimizer] = []
  if matrix_params:
    optimizers.append(
      NorMuon(
        matrix_params,
        lr=learning_rate,
        momentum=momentum,
        beta2=beta2,
        weight_decay=weight_decay,
      )
    )
  if adamw_params:
    optimizers.append(
      CautiousAdamW(
        [{"params": adamw_params, "weight_decay": weight_decay}],
        lr=learning_rate,
        betas=adamw_betas,
        eps=adamw_eps,
      )
    )
  return CompositeOptimizer(optimizers)


def configure_speedrun_normuon(
  model: nn.Module,
  *,
  adam_learning_rate: float = 0.008,
  matrix_learning_rate: float = 0.023,
  adam_weight_decay: float = 0.005,
  matrix_weight_decay: float = 1.2,
  momentum: float = 0.95,
  beta2: float = 0.9,
  adam_eps: float = 1e-10,
) -> CompositeOptimizer:
  """Configure the current short-track per-parameter optimizer recipe."""

  matrix_groups = []
  adam_groups = []
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if _use_matrix_optimizer(name, param):
      lr_scale = 2.0 if name.endswith("ffn.down.weight") else 1.0
      matrix_groups.append(
        {
          "params": [param],
          "lr": matrix_learning_rate * lr_scale,
          "weight_decay": matrix_weight_decay,
          "name": name,
        }
      )
      continue
    lr_scale, wd_scale, betas = _speedrun_adam_settings(name)
    adam_groups.append(
      {
        "params": [param],
        "lr": adam_learning_rate * lr_scale,
        "weight_decay": adam_weight_decay * wd_scale,
        "betas": betas,
        "name": name,
      }
    )

  optimizers: list[torch.optim.Optimizer] = []
  if matrix_groups:
    optimizers.append(
      NorMuon(
        matrix_groups,
        lr=matrix_learning_rate,
        momentum=momentum,
        beta2=beta2,
        weight_decay=matrix_weight_decay,
      )
    )
  if adam_groups:
    optimizers.append(
      CautiousAdamW(
        adam_groups,
        lr=adam_learning_rate,
        eps=adam_eps,
        weight_decay=adam_weight_decay,
      )
    )
  return CompositeOptimizer(optimizers)


def _use_matrix_optimizer(name: str, param: nn.Parameter) -> bool:
  if param.ndim != 2:
    return False
  if any(token in name for token in ("embedding", "lm_head", "gate", "norm")):
    return False
  if name in {
    "residual_scales",
    "post_scales",
    "xsa_alphas",
  }:
    return False
  return not (name.startswith("mudd.") or name.endswith("value_embeddings"))


def _speedrun_adam_settings(
  name: str,
) -> tuple[float, float, tuple[float, float]]:
  if name in {"token_embedding.weight", "lm_head.weight"}:
    return 1.0, 150.0, (0.5, 0.95)
  if name == "bigram_embedding.embedding.weight":
    return 75.0, 5.0, (0.75, 0.95)
  if name == "value_embeddings":
    return 75.0, 5.0, (0.75, 0.95)
  if name == "token_smear.gate.weight":
    return 0.01, 0.0, (0.9, 0.99)
  if name == "block_skip_gate.weight":
    return 0.05, 0.0, (0.9, 0.99)
  if name.startswith("mudd."):
    wd_scale = 0.0 if name == "mudd.bias" else 1.0
    return 0.25, wd_scale, (0.9, 0.99)
  if name in {"attention_gate_weights", "value_gate_weights"}:
    return 1.0, 1.0, (0.9, 0.99)
  if name in {"residual_scales", "token_smear.scale", "block_skip_logit"}:
    return 5.0, 0.0, (0.9, 0.95)
  if name in {
    "post_scales",
    "embedding_skip_weights",
    "bigram_injection_weights",
    "value_mix_logits",
    "xsa_alphas",
  }:
    return 1.0, 0.0, (0.9, 0.95)
  return 1.0, 1.0, (0.9, 0.99)


def _orthogonalize_update(update: torch.Tensor, *, ns_steps: int) -> torch.Tensor:
  original_shape = update.shape
  matrix = update.flatten(1) if update.ndim > 2 else update
  orthogonalized = _zeroth_power_via_newton_schulz5(matrix, steps=ns_steps)
  return orthogonalized.reshape(original_shape)


def _zeroth_power_via_newton_schulz5(
  matrix: torch.Tensor, *, steps: int
) -> torch.Tensor:
  if matrix.ndim != 2:
    raise ValueError("Newton-Schulz orthogonalization expects a matrix")

  a, b, c = 3.4445, -4.7750, 2.0315
  original_dtype = matrix.dtype
  x = matrix.bfloat16()
  if x.size(0) > x.size(1):
    x = x.T

  x.div_(x.norm().clamp(min=1e-7))
  for _ in range(steps):
    xx_t = x @ x.T
    gram_update = torch.addmm(xx_t, xx_t, xx_t, beta=b, alpha=c)
    x = torch.addmm(x, gram_update, x, beta=a)

  if matrix.size(0) > matrix.size(1):
    x = x.T
  return x.to(dtype=original_dtype)


_POLAR_EXPRESS_COEFFICIENTS = (
  (8.156554524902461, -22.48329292557795, 15.878769915207462),
  (4.042929935166739, -2.808917465908714, 0.5000178451051316),
  (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
  (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
  (2.3465413258596377, -1.709782838708108, 0.42323551169305323),
)


def _polar_express(update: torch.Tensor) -> torch.Tensor:
  original_dtype = update.dtype
  x = update.bfloat16()
  transposed = x.shape[-2] > x.shape[-1]
  if transposed:
    x = x.T
  x = x / (x.norm() * 1.02 + 1e-6)
  for a, b, c in _POLAR_EXPRESS_COEFFICIENTS:
    gram = x @ x.T
    x = a * x + (b * gram + c * gram @ gram) @ x
  if transposed:
    x = x.T
  return x.to(original_dtype)


def _normuon_variance_reduction(
  update: torch.Tensor,
  second_momentum: torch.Tensor,
  *,
  beta2: float,
  red_dim: int,
) -> torch.Tensor:
  variance = update.float().square().mean(dim=red_dim, keepdim=True)
  red_dim_size = update.size(red_dim)
  original_norm = (variance.sum() * red_dim_size).sqrt()
  second_momentum.lerp_(variance, 1.0 - beta2)
  scale = second_momentum.clamp_min(1e-10).rsqrt()
  scaled_norm = (variance * red_dim_size * scale.float().square()).sum().sqrt()
  return update * scale * (original_norm / scaled_norm.clamp_min(1e-10))


def _adjust_muon_lr(lr: float, shape: torch.Size) -> float:
  rows = shape[0]
  cols = math.prod(shape[1:])
  return lr * max(1.0, rows / cols) ** 0.5
