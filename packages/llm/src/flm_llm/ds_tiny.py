"""DS-Tiny decoder-only language model."""

from __future__ import annotations

import torch
from flm_modules import DeepSeekMLA, RMSNorm, RopeLayout, SwiGLU
from torch import nn

from flm_llm.config import DSTinyConfig
from flm_llm.losses import language_model_loss


class DSTinyBlock(nn.Module):
  def __init__(self, config: DSTinyConfig) -> None:
    super().__init__()
    self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.attn = DeepSeekMLA(
      d_model=config.d_model,
      n_heads=config.n_heads,
      kv_lora_rank=config.kv_lora_rank,
      q_lora_rank=config.attention_q_lora_rank,
      qk_nope_head_dim=config.qk_nope_head_dim,
      qk_rope_head_dim=config.qk_rope_head_dim,
      v_head_dim=config.v_head_dim,
      bias=config.bias,
      rope_base=config.rope_base,
      rope_layout=RopeLayout.DEEPSEEK_V32,
      norm_eps=config.norm_eps,
      backend=config.attention_backend,
    )
    self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.ffn = SwiGLU(
      d_model=config.d_model,
      d_ff=config.ffn_d_ff,
      bias=config.bias,
    )

  def forward(
    self,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
  ) -> torch.Tensor:
    x = x + self.attn(
      self.attn_norm(x),
      attention_mask=attention_mask,
      positions=positions,
    )
    return x + self.ffn(self.ffn_norm(x))


class DSTiny(nn.Module):
  def __init__(self, config: DSTinyConfig) -> None:
    super().__init__()
    self.config = config
    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    self.blocks = nn.ModuleList(DSTinyBlock(config) for _ in range(config.n_layers))
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
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    attention_mask = _causal_mask(
      batch_size=input_ids.shape[0],
      seq_len=input_ids.shape[1],
      dtype=x.dtype,
      device=x.device,
    )
    for block in self.blocks:
      x = block(x, attention_mask=attention_mask, positions=positions)
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


def _causal_mask(
  batch_size: int,
  seq_len: int,
  dtype: torch.dtype,
  device: torch.device,
) -> torch.Tensor:
  mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
  mask = torch.triu(mask, diagonal=1)
  return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1)
