"""Multi-head latent attention layers."""

from __future__ import annotations

import torch
from torch import nn

from flm_modules.attentions.backends import (
  AttentionBackend,
  scaled_dot_product_attention,
)
from flm_modules.norm import RMSNorm
from flm_modules.rope import RotaryEmbedding


class DeepSeekMLA(nn.Module):
  """DeepSeek-style multi-head latent attention."""

  def __init__(
    self,
    d_model: int,
    n_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int | None = None,
    bias: bool = False,
    rope_base: float = 10_000.0,
    rope_layout: str = "llama",
    norm_eps: float = 1e-6,
    causal: bool = True,
    backend: AttentionBackend | str = AttentionBackend.TORCH,
  ) -> None:
    super().__init__()
    if n_heads <= 0:
      raise ValueError("n_heads must be positive")
    if kv_lora_rank <= 0:
      raise ValueError("kv_lora_rank must be positive")
    if q_lora_rank is not None and q_lora_rank <= 0:
      raise ValueError("q_lora_rank must be positive when provided")
    if qk_rope_head_dim <= 0 or qk_rope_head_dim % 2 != 0:
      raise ValueError("qk_rope_head_dim must be a positive even number")
    if qk_nope_head_dim <= 0:
      raise ValueError("qk_nope_head_dim must be positive")
    if v_head_dim <= 0:
      raise ValueError("v_head_dim must be positive")

    self.d_model = d_model
    self.n_heads = n_heads
    self.kv_lora_rank = kv_lora_rank
    self.q_lora_rank = q_lora_rank
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.scaling = self.qk_head_dim**-0.5
    self.causal = causal
    self.backend = AttentionBackend(backend)

    if q_lora_rank is None:
      self.q_proj = nn.Linear(d_model, n_heads * self.qk_head_dim, bias=False)
    else:
      self.q_a_proj = nn.Linear(d_model, q_lora_rank, bias=bias)
      self.q_a_layernorm = RMSNorm(q_lora_rank, eps=norm_eps)
      self.q_b_proj = nn.Linear(q_lora_rank, n_heads * self.qk_head_dim, bias=False)

    self.kv_a_proj_with_mqa = nn.Linear(
      d_model,
      kv_lora_rank + qk_rope_head_dim,
      bias=bias,
    )
    self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=norm_eps)
    self.kv_b_proj = nn.Linear(
      kv_lora_rank,
      n_heads * (qk_nope_head_dim + v_head_dim),
      bias=False,
    )
    self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=bias)
    self.rope = RotaryEmbedding(
      qk_rope_head_dim,
      base=rope_base,
      layout=rope_layout,
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    query_states, key_states, value_states = self._project_qkv(
      hidden_states,
      positions=positions,
    )
    attn_output = scaled_dot_product_attention(
      query_states,
      key_states,
      value_states,
      backend=self.backend,
      attn_mask=attention_mask,
      causal=self.causal,
      scale=self.scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.v_head_dim)
    return self.o_proj(attn_output)

  def _project_qkv(
    self,
    hidden_states: torch.Tensor,
    positions: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = hidden_states.shape
    q_states = self._project_q(hidden_states)
    q_states = q_states.view(
      batch_size,
      seq_len,
      self.n_heads,
      self.qk_head_dim,
    ).transpose(1, 2)
    q_pass, q_rot = torch.split(
      q_states,
      [self.qk_nope_head_dim, self.qk_rope_head_dim],
      dim=-1,
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(
      compressed_kv,
      [self.kv_lora_rank, self.qk_rope_head_dim],
      dim=-1,
    )
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass))
    k_pass = k_pass.view(
      batch_size,
      seq_len,
      self.n_heads,
      self.qk_nope_head_dim + self.v_head_dim,
    ).transpose(1, 2)
    k_pass, value_states = torch.split(
      k_pass,
      [self.qk_nope_head_dim, self.v_head_dim],
      dim=-1,
    )

    k_rot = k_rot.view(batch_size, 1, seq_len, self.qk_rope_head_dim)
    q_rot, k_rot = self.rope(q_rot, k_rot, positions=positions)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    return query_states, key_states, value_states

  def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.q_lora_rank is None:
      return self.q_proj(hidden_states)
    return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
