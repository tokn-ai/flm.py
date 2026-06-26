"""DeepSeek Sparse Attention layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from flm_modules.attentions.backends import (
  AttentionBackend,
  scaled_dot_product_attention,
)
from flm_modules.norm import RMSNorm
from flm_modules.rope import RopeLayout, RotaryEmbedding, apply_rotary


class DeepSeekDSAIndexer(nn.Module):
  """DeepSeek V3.2 sparse attention top-k token indexer."""

  def __init__(
    self,
    d_model: int,
    q_lora_rank: int,
    qk_rope_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope_base: float = 10_000.0,
  ) -> None:
    super().__init__()
    if d_model <= 0:
      raise ValueError("d_model must be positive")
    if q_lora_rank <= 0:
      raise ValueError("q_lora_rank must be positive")
    if qk_rope_head_dim <= 0 or qk_rope_head_dim % 2 != 0:
      raise ValueError("qk_rope_head_dim must be a positive even number")
    if index_n_heads <= 0:
      raise ValueError("index_n_heads must be positive")
    if index_head_dim <= 0:
      raise ValueError("index_head_dim must be positive")
    if qk_rope_head_dim > index_head_dim:
      raise ValueError("qk_rope_head_dim must not exceed index_head_dim")
    if index_topk <= 0:
      raise ValueError("index_topk must be positive")

    self.d_model = d_model
    self.q_lora_rank = q_lora_rank
    self.qk_rope_head_dim = qk_rope_head_dim
    self.index_n_heads = index_n_heads
    self.index_head_dim = index_head_dim
    self.index_topk = index_topk
    self.softmax_scale = index_head_dim**-0.5
    self.weights_scaling = index_n_heads**-0.5

    self.wq_b = nn.Linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
    self.wk = nn.Linear(d_model, index_head_dim, bias=False)
    self.k_norm = nn.LayerNorm(index_head_dim, eps=1e-6)
    self.weights_proj = nn.Linear(d_model, index_n_heads, bias=False)
    self.rope = RotaryEmbedding(
      qk_rope_head_dim,
      base=rope_base,
      layout=RopeLayout.LLAMA,
    )

  @torch.no_grad()
  def forward(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    if positions is None:
      positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    if positions.ndim == 1:
      positions = positions.unsqueeze(0)
    positions = positions.expand(batch_size, -1)

    q = self.wq_b(q_residual)
    q = q.view(batch_size, seq_len, self.index_n_heads, self.index_head_dim)
    q_rot, q_pass = torch.split(
      q,
      [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
      dim=-1,
    )

    k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(
      k,
      [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
      dim=-1,
    )

    cos, sin = self.rope(q_rot, positions=positions)
    q_rot = apply_rotary(
      q_rot,
      cos.unsqueeze(2),
      sin.unsqueeze(2),
      layout=RopeLayout.LLAMA,
    )
    k_rot = apply_rotary(
      k_rot,
      cos.unsqueeze(2),
      sin.unsqueeze(2),
      layout=RopeLayout.LLAMA,
    )
    q = torch.cat((q_rot, q_pass), dim=-1)
    k = torch.cat((k_rot, k_pass), dim=-1).squeeze(2)

    scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores * self.softmax_scale)
    weights = self.weights_proj(hidden_states).float() * self.weights_scaling
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    if attention_mask is not None:
      index_scores = index_scores + attention_mask
    else:
      key_positions = torch.arange(index_scores.shape[-1], device=index_scores.device)
      causal = key_positions.view(1, 1, -1) > positions.unsqueeze(-1)
      index_scores = index_scores.masked_fill(causal, float("-inf"))

    topk = min(self.index_topk, index_scores.shape[-1])
    return index_scores.topk(topk, dim=-1).indices.to(torch.int32)


class DeepSeekDSA(nn.Module):
  """DeepSeek V3.2 MLA attention with DSA sparse token selection."""

  def __init__(
    self,
    d_model: int,
    n_heads: int,
    kv_lora_rank: int,
    q_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    bias: bool = False,
    rope_base: float = 10_000.0,
    norm_eps: float = 1e-6,
    causal: bool = True,
    backend: AttentionBackend | str = AttentionBackend.TORCH,
  ) -> None:
    super().__init__()
    if n_heads <= 0:
      raise ValueError("n_heads must be positive")
    if kv_lora_rank <= 0:
      raise ValueError("kv_lora_rank must be positive")
    if q_lora_rank <= 0:
      raise ValueError("q_lora_rank must be positive")
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
      layout=RopeLayout.LLAMA,
    )
    self.indexer = DeepSeekDSAIndexer(
      d_model=d_model,
      q_lora_rank=q_lora_rank,
      qk_rope_head_dim=qk_rope_head_dim,
      index_n_heads=index_n_heads,
      index_head_dim=index_head_dim,
      index_topk=index_topk,
      rope_base=rope_base,
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    if positions is None:
      positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    if positions.ndim == 1:
      positions = positions.unsqueeze(0)
    positions = positions.expand(batch_size, -1)

    query_states, key_states, value_states, q_residual = self._project_qkv(
      hidden_states,
      positions=positions,
    )
    indexer_mask = attention_mask[:, 0] if attention_mask is not None else None
    topk_indices = self.indexer(
      hidden_states,
      q_residual,
      attention_mask=indexer_mask,
      positions=positions,
    )
    sparse_mask = topk_indices.new_ones(
      (batch_size, seq_len, key_states.shape[2]),
      dtype=torch.bool,
    )
    sparse_mask = sparse_mask.scatter(-1, topk_indices.long(), False).unsqueeze(1)
    if attention_mask is None:
      key_positions = torch.arange(key_states.shape[2], device=hidden_states.device)
      sparse_mask = sparse_mask | (
        key_positions.view(1, 1, 1, -1) > positions.unsqueeze(1).unsqueeze(-1)
      )
      attention_mask = hidden_states.new_zeros(
        (batch_size, 1, seq_len, key_states.shape[2]),
      )
    attention_mask = attention_mask.masked_fill(
      sparse_mask,
      torch.finfo(hidden_states.dtype).min,
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
    positions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = hidden_states.shape
    q_residual = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_states = self.q_b_proj(q_residual)
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
    cos, sin = self.rope(q_rot, positions=positions)
    q_rot, k_rot = _apply_interleave_rotary(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    return query_states, key_states, value_states, q_residual


def _apply_interleave_rotary(
  q: torch.Tensor,
  k: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  cos = cos[..., : cos.shape[-1] // 2].unsqueeze(1)
  sin = sin[..., : sin.shape[-1] // 2].unsqueeze(1)

  q1, q2 = q[..., 0::2], q[..., 1::2]
  k1, k2 = k[..., 0::2], k[..., 1::2]

  q_embed = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
  k_embed = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
  return q_embed, k_embed
