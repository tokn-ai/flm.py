"""Eager reference model for the nanoGPT short-track speedrun architecture."""

from __future__ import annotations

import torch
from flm_modules import QKNormSelfAttention, ReLUSquared, RMSNorm
from flm_modules.losses import language_model_loss
from torch import nn
from torch.nn import functional as F

from flm_llm.config import NanoGPTSpeedrunConfig


class NanoGPTSpeedrunBlock(nn.Module):
  def __init__(self, config: NanoGPTSpeedrunConfig) -> None:
    super().__init__()
    self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.attn = QKNormSelfAttention(
      d_model=config.d_model,
      n_heads=config.n_heads,
      bias=config.bias,
      rope_base=config.rope_base,
      norm_eps=config.norm_eps,
      backend=config.attention_backend,
      zero_init_out=True,
    )
    self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.ffn = ReLUSquared(
      d_model=config.d_model,
      d_ff=config.d_ff,
      bias=config.bias,
      zero_init_down=True,
    )
    self.residual_decay = config.residual_decay

  def forward(
    self,
    x: torch.Tensor,
    *,
    value_residual: torch.Tensor | None = None,
    value_mix: torch.Tensor | float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    attn_output, values = self.attn(
      self.attn_norm(x),
      value_residual=value_residual,
      value_mix=value_mix,
    )
    x = self.residual_decay * x + attn_output
    x = self.residual_decay * x + self.ffn(self.ffn_norm(x))
    return x, values


class NanoGPTSpeedrunModel(nn.Module):
  """Portable semantic baseline for current nanoGPT speedrun features.

  This intentionally uses ordinary PyTorch operations. Distributed optimizer
  sharding, FP8, FlashAttention 3, and fused H100 kernels belong to a later
  execution backend and do not change this model's public contract.
  """

  def __init__(self, config: NanoGPTSpeedrunConfig) -> None:
    super().__init__()
    if config.n_layers < 1:
      raise ValueError("n_layers must be positive")
    if config.logit_softcap is not None and config.logit_softcap <= 0:
      raise ValueError("logit_softcap must be positive")
    if (config.block_skip_from is None) != (config.block_skip_to is None):
      raise ValueError("block skip endpoints must both be set or both be None")
    if config.block_skip_from is not None:
      if not 0 <= config.block_skip_from < config.block_skip_to < config.n_layers:
        raise ValueError("block skip endpoints must be ordered layer indices")

    self.config = config
    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
    self.blocks = nn.ModuleList(
      NanoGPTSpeedrunBlock(config) for _ in range(config.n_layers)
    )
    self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    if config.tie_embeddings:
      self.lm_head.weight = self.token_embedding.weight

    if config.embedding_skip:
      self.embedding_skip_weights = nn.Parameter(
        torch.full((config.n_layers,), config.n_layers**-0.5)
      )
    else:
      self.register_parameter("embedding_skip_weights", None)
    if config.value_residual and config.n_layers > 1:
      self.value_mix_logits = nn.Parameter(torch.zeros(config.n_layers - 1))
    else:
      self.register_parameter("value_mix_logits", None)
    if config.block_skip_from is not None:
      self.block_skip_weight = nn.Parameter(torch.ones(()))
    else:
      self.register_parameter("block_skip_weight", None)

  def forward(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    return_logits: bool = True,
  ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if input_ids.ndim != 2:
      raise ValueError("input_ids must have shape (batch, seq_len)")
    if input_ids.shape[1] > self.config.max_seq_len:
      raise ValueError("sequence length exceeds config.max_seq_len")

    embeddings = self.token_embedding(input_ids)
    x = embeddings
    first_values = None
    block_skip = None
    for layer_index, block in enumerate(self.blocks):
      if self.embedding_skip_weights is not None:
        x = x + self.embedding_skip_weights[layer_index] * embeddings
      if layer_index == self.config.block_skip_to:
        if block_skip is None or self.block_skip_weight is None:
          raise RuntimeError("block skip source was not captured")
        x = x + self.block_skip_weight * block_skip

      value_mix = None
      if first_values is not None and self.value_mix_logits is not None:
        value_mix = self.value_mix_logits[layer_index - 1].sigmoid()
      x, values = block(
        x,
        value_residual=first_values,
        value_mix=value_mix,
      )
      if first_values is None and self.config.value_residual:
        first_values = values
      if layer_index == self.config.block_skip_from:
        block_skip = x

    hidden_states = self.norm(x)
    needs_logits_for_loss = (
      targets is not None and self.config.logit_softcap is not None
    )
    logits = (
      self._logits(hidden_states)
      if return_logits or needs_logits_for_loss
      else None
    )
    loss = self._loss(hidden_states, logits, targets)
    return logits if return_logits else None, loss

  def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
    logits = self.config.logit_scale * self.lm_head(hidden_states)
    if self.config.logit_softcap is not None:
      cap = self.config.logit_softcap
      logits = cap * torch.tanh(logits / cap)
    return logits

  def _loss(
    self,
    hidden_states: torch.Tensor,
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
  ) -> torch.Tensor | None:
    if targets is None:
      return None
    if self.config.logit_softcap is not None:
      if logits is None:
        raise RuntimeError("softcapped loss requires logits")
      return F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    return language_model_loss(
      hidden_states=hidden_states,
      classifier_weight=self.lm_head.weight,
      targets=targets,
      backend=self.config.loss_backend,
      chunk_size=self.config.loss_chunk_size,
    )
