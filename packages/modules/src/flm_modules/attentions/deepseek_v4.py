"""DeepSeek V4 attention layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from flm_modules.hyper import UnweightedRMSNorm
from flm_modules.linear import GroupedLinear
from flm_modules.norm import RMSNorm
from flm_modules.rope import RopeLayout, rotate_half


class DeepSeekV4RotaryEmbedding(nn.Module):
  def __init__(
    self,
    head_dim: int,
    rope_head_dim: int,
    base: float = 10_000.0,
  ) -> None:
    super().__init__()
    if head_dim <= 0:
      raise ValueError("head_dim must be positive")
    if rope_head_dim <= 0 or rope_head_dim % 2 != 0:
      raise ValueError("rope_head_dim must be a positive even number")
    if rope_head_dim > head_dim:
      raise ValueError("rope_head_dim must not exceed head_dim")
    self.head_dim = head_dim
    self.rope_head_dim = rope_head_dim
    inv_freq = 1.0 / (
      base ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32) / rope_head_dim)
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def forward(
    self,
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = x.shape[:2]
    if positions is None:
      positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
    if positions.ndim == 1:
      positions = positions.unsqueeze(0)
    positions = positions.expand(batch_size, -1)

    freqs = positions.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
    cos = freqs.cos().to(dtype=x.dtype)
    sin = freqs.sin().to(dtype=x.dtype)
    return cos, sin


def apply_deepseek_v4_rotary(
  x: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  *,
  unsqueeze_dim: int = 1,
) -> torch.Tensor:
  cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
  sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
  rope_dim = cos.shape[-1]
  nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
  rotated = (
    rope.float() * cos + rotate_half(rope, layout=RopeLayout.INTERLEAVED).float() * sin
  ).to(x.dtype)
  return torch.cat((nope, rotated), dim=-1)


class DeepSeekV4Attention(nn.Module):
  """Sliding DeepSeek V4 attention with shared K/V and attention sinks."""

  def __init__(
    self,
    d_model: int,
    n_heads: int,
    head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    o_groups: int,
    rope_head_dim: int | None = None,
    bias: bool = False,
    rope_base: float = 10_000.0,
    norm_eps: float = 1e-6,
  ) -> None:
    super().__init__()
    if d_model <= 0:
      raise ValueError("d_model must be positive")
    if n_heads <= 0:
      raise ValueError("n_heads must be positive")
    if head_dim <= 0:
      raise ValueError("head_dim must be positive")
    if q_lora_rank <= 0:
      raise ValueError("q_lora_rank must be positive")
    if o_lora_rank <= 0:
      raise ValueError("o_lora_rank must be positive")
    if o_groups <= 0 or (n_heads * head_dim) % o_groups != 0:
      raise ValueError("o_groups must divide n_heads * head_dim")

    rope_head_dim = rope_head_dim if rope_head_dim is not None else head_dim
    self.d_model = d_model
    self.n_heads = n_heads
    self.head_dim = head_dim
    self.q_lora_rank = q_lora_rank
    self.o_lora_rank = o_lora_rank
    self.o_groups = o_groups
    self.rope_head_dim = rope_head_dim
    self.scaling = head_dim**-0.5

    self.q_a_proj = nn.Linear(d_model, q_lora_rank, bias=bias)
    self.q_a_norm = RMSNorm(q_lora_rank, eps=norm_eps)
    self.q_b_proj = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
    self.q_b_norm = UnweightedRMSNorm(eps=norm_eps)
    self.kv_proj = nn.Linear(d_model, head_dim, bias=bias)
    self.kv_norm = RMSNorm(head_dim, eps=norm_eps)
    self.o_a_proj = GroupedLinear(
      n_heads * head_dim // o_groups,
      o_groups * o_lora_rank,
      o_groups,
      bias=False,
    )
    self.o_b_proj = nn.Linear(o_groups * o_lora_rank, d_model, bias=False)
    self.sinks = nn.Parameter(torch.zeros(n_heads))
    self.rope = DeepSeekV4RotaryEmbedding(
      head_dim=head_dim,
      rope_head_dim=rope_head_dim,
      base=rope_base,
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, self.n_heads, self.head_dim)
    cos, sin = self.rope(hidden_states, positions=positions)

    q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
    query_states = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
    query_states = self.q_b_norm(query_states)
    query_states = apply_deepseek_v4_rotary(query_states, cos, sin)

    kv_states = self.kv_norm(self.kv_proj(hidden_states))
    kv_states = kv_states.view(*input_shape, 1, self.head_dim).transpose(1, 2)
    kv_states = apply_deepseek_v4_rotary(kv_states, cos, sin)

    attn_output = self._attention(
      query_states,
      kv_states,
      attention_mask=attention_mask,
    )
    attn_output = apply_deepseek_v4_rotary(
      attn_output.transpose(1, 2),
      cos,
      -sin,
    ).transpose(1, 2)

    grouped = attn_output.reshape(*input_shape, self.o_groups, -1)
    grouped = self.o_a_proj(grouped).flatten(2)
    return self.o_b_proj(grouped)

  def _attention(
    self,
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
  ) -> torch.Tensor:
    key_states = kv_states.expand(-1, self.n_heads, -1, -1)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
      attn_weights = attn_weights + attention_mask

    sinks = self.sinks.reshape(1, -1, 1, 1).expand(
      query_states.shape[0],
      -1,
      query_states.shape[-2],
      -1,
    )
    combined_logits = torch.cat((attn_weights, sinks), dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    attn_weights = probs[..., :-1].to(kv_states.dtype)
    return torch.matmul(attn_weights, key_states).transpose(1, 2).contiguous()
