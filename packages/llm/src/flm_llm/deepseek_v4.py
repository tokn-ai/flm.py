"""DeepSeekV4 decoder-only language model."""

from __future__ import annotations

import torch
from flm_modules import (
  DeepSeekMoE,
  DeepSeekV4Attention,
  DeepSeekV4HyperConnection,
  DeepSeekV4HyperHead,
  ExpertKind,
  RMSNorm,
  RouterScoring,
  SwiGLU,
)
from torch import nn
from torch.nn import functional as F

from flm_llm.config import DeepSeekV4Config


class DeepSeekV4Block(nn.Module):
  def __init__(self, config: DeepSeekV4Config, layer_idx: int) -> None:
    super().__init__()
    self.attn_hc = DeepSeekV4HyperConnection(
      d_model=config.d_model,
      hc_mult=config.hc_mult,
      hc_sinkhorn_iters=config.hc_sinkhorn_iters,
      hc_eps=config.hc_eps,
      rms_norm_eps=config.norm_eps,
      initializer_range=config.initializer_range,
    )
    self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.attn = DeepSeekV4Attention(
      d_model=config.d_model,
      n_heads=config.n_heads,
      head_dim=config.attention_head_dim,
      q_lora_rank=config.attention_q_lora_rank,
      o_lora_rank=config.attention_o_lora_rank,
      o_groups=config.o_groups,
      rope_head_dim=config.attention_rope_head_dim,
      bias=config.bias,
      rope_base=config.rope_base,
      norm_eps=config.norm_eps,
      layer_type=config.attention_layer_type(layer_idx),
      compress_rate_csa=config.compress_rate_csa,
      compress_rate_hca=config.compress_rate_hca,
      index_n_heads=config.index_n_heads,
      index_head_dim=config.index_head_dim,
      index_topk=config.index_topk,
    )
    self.ffn_hc = DeepSeekV4HyperConnection(
      d_model=config.d_model,
      hc_mult=config.hc_mult,
      hc_sinkhorn_iters=config.hc_sinkhorn_iters,
      hc_eps=config.hc_eps,
      rms_norm_eps=config.norm_eps,
      initializer_range=config.initializer_range,
    )
    self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    if layer_idx < config.dense_layers:
      self.ffn = SwiGLU(
        d_model=config.d_model,
        d_ff=config.ffn_d_ff,
        bias=config.bias,
      )
    else:
      self.ffn = DeepSeekMoE(
        d_model=config.d_model,
        d_ff=config.ffn_d_ff,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_experts_per_token=config.n_experts_per_token,
        n_group=config.n_group,
        topk_group=config.topk_group,
        norm_topk_prob=config.norm_topk_prob,
        routed_scaling_factor=config.routed_scaling_factor,
        bias=config.bias,
        scoring_func=RouterScoring.SQRT_SOFTPLUS,
        grouped_topk=False,
        expert_kind=ExpertKind.V4,
      )

  def forward(
    self,
    hidden_streams: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
  ) -> torch.Tensor:
    dtype = hidden_streams.dtype
    post, comb, collapsed = self.attn_hc(hidden_streams)
    attn_output = self.attn(
      self.attn_norm(collapsed),
      attention_mask=attention_mask,
      positions=positions,
    )
    hidden_streams = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(
      -2
    ) + torch.matmul(
      comb.to(dtype).transpose(-1, -2),
      hidden_streams,
    )

    post, comb, collapsed = self.ffn_hc(hidden_streams)
    ffn_output = self.ffn(self.ffn_norm(collapsed))
    return post.to(dtype).unsqueeze(-1) * ffn_output.unsqueeze(-2) + torch.matmul(
      comb.to(dtype).transpose(-1, -2),
      hidden_streams,
    )


class DeepSeekV4(nn.Module):
  def __init__(self, config: DeepSeekV4Config) -> None:
    super().__init__()
    self.config = config
    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    self.blocks = nn.ModuleList(
      DeepSeekV4Block(config, layer_idx) for layer_idx in range(config.n_layers)
    )
    self.hc_head = DeepSeekV4HyperHead(
      d_model=config.d_model,
      hc_mult=config.hc_mult,
      hc_eps=config.hc_eps,
      rms_norm_eps=config.norm_eps,
      initializer_range=config.initializer_range,
    )
    self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    self.lm_head.weight = self.token_embedding.weight

  def forward(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    if input_ids.ndim != 2:
      raise ValueError("input_ids must have shape (batch, seq_len)")
    if input_ids.shape[1] > self.config.max_seq_len:
      raise ValueError("sequence length exceeds config.max_seq_len")

    x = self.token_embedding(input_ids)
    x = x.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    attention_mask = _causal_mask(
      batch_size=input_ids.shape[0],
      seq_len=input_ids.shape[1],
      dtype=x.dtype,
      device=x.device,
    )
    for block in self.blocks:
      x = block(x, attention_mask=attention_mask, positions=positions)
    logits = self.lm_head(self.norm(self.hc_head(x)))

    loss = None
    if targets is not None:
      loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
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
