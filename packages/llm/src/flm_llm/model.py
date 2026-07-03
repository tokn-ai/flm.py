"""Reference decoder-only language model."""

from __future__ import annotations

import torch
from flm_modules import RMSNorm, SelfAttention, SwiGLU
from flm_modules.losses import language_model_loss
from torch import nn

from flm_llm.config import ReferenceModelConfig


class TransformerBlock(nn.Module):
  def __init__(self, config: ReferenceModelConfig) -> None:
    super().__init__()
    self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.attn = SelfAttention(
      d_model=config.d_model,
      n_heads=config.n_heads,
      bias=config.bias,
      rope_base=config.rope_base,
      backend=config.attention_backend,
    )
    self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.ffn = SwiGLU(
      d_model=config.d_model,
      d_ff=config.ffn_d_ff,
      bias=config.bias,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.attn_norm(x))
    return x + self.ffn(self.ffn_norm(x))


class ReferenceModel(nn.Module):
  def __init__(self, config: ReferenceModelConfig) -> None:
    super().__init__()
    self.config = config
    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    self.blocks = nn.ModuleList(
      TransformerBlock(config) for _ in range(config.n_layers)
    )
    self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    self.lm_head.weight = self.token_embedding.weight

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

    x = self.token_embedding(input_ids)
    for block in self.blocks:
      x = block(x)
    hidden_states = self.norm(x)
    logits = self.lm_head(hidden_states) if return_logits else None

    loss = None
    if targets is not None:
      loss = language_model_loss(
        hidden_states=hidden_states,
        classifier_weight=self.lm_head.weight,
        targets=targets,
        backend=self.config.loss_backend,
        chunk_size=self.config.loss_chunk_size,
      )
    return logits, loss
