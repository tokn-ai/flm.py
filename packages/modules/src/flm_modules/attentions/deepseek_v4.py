"""DeepSeek V4 attention layers."""

from __future__ import annotations

from enum import StrEnum

import torch
from torch import nn
from torch.nn import functional as F

from flm_modules.hyper import UnweightedRMSNorm
from flm_modules.linear import GroupedLinear
from flm_modules.norm import RMSNorm
from flm_modules.rope import RopeLayout, rotate_half


class DeepSeekV4AttentionKind(StrEnum):
  SLIDING = "sliding_attention"
  COMPRESSED_SPARSE = "compressed_sparse_attention"
  HEAVILY_COMPRESSED = "heavily_compressed_attention"


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
  """DeepSeek V4 attention with optional compressed K/V blocks."""

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
    layer_type: DeepSeekV4AttentionKind | str = DeepSeekV4AttentionKind.SLIDING,
    compress_rate_csa: int = 4,
    compress_rate_hca: int = 128,
    index_n_heads: int = 64,
    index_head_dim: int = 128,
    index_topk: int = 512,
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
    self.layer_type = DeepSeekV4AttentionKind(layer_type)

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
    self.compressor: DeepSeekV4CSACompressor | DeepSeekV4HCACompressor | None
    if self.layer_type == DeepSeekV4AttentionKind.COMPRESSED_SPARSE:
      self.compressor = DeepSeekV4CSACompressor(
        d_model=d_model,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        compress_rate=compress_rate_csa,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        rope_head_dim=rope_head_dim,
        rope_base=rope_base,
        norm_eps=norm_eps,
      )
    elif self.layer_type == DeepSeekV4AttentionKind.HEAVILY_COMPRESSED:
      self.compressor = DeepSeekV4HCACompressor(
        d_model=d_model,
        head_dim=head_dim,
        compress_rate=compress_rate_hca,
        rope_head_dim=rope_head_dim,
        rope_base=rope_base,
        norm_eps=norm_eps,
      )
    else:
      self.compressor = None

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

    if self.compressor is not None:
      if positions is None:
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
      if self.layer_type == DeepSeekV4AttentionKind.COMPRESSED_SPARSE:
        compressed_kv, block_bias = self.compressor(
          hidden_states,
          q_residual,
          positions,
        )
      else:
        compressed_kv, block_bias = self.compressor(hidden_states, positions)
      kv_states = torch.cat((kv_states, compressed_kv), dim=2)
      if attention_mask is not None and block_bias is not None:
        attention_mask = torch.cat(
          (attention_mask, block_bias.to(attention_mask.dtype)),
          dim=-1,
        )

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


class DeepSeekV4IndexerScorer(nn.Module):
  """Lightning-indexer scoring head for compressed sparse attention."""

  def __init__(
    self,
    d_model: int,
    index_n_heads: int,
    index_head_dim: int,
  ) -> None:
    super().__init__()
    if d_model <= 0:
      raise ValueError("d_model must be positive")
    if index_n_heads <= 0:
      raise ValueError("index_n_heads must be positive")
    if index_head_dim <= 0:
      raise ValueError("index_head_dim must be positive")

    self.softmax_scale = index_head_dim**-0.5
    self.weights_scaling = index_n_heads**-0.5
    self.weights_proj = nn.Linear(d_model, index_n_heads, bias=False)

  def forward(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    hidden_states: torch.Tensor,
  ) -> torch.Tensor:
    scores = torch.matmul(
      q.float(),
      compressed_kv.transpose(-1, -2).float().unsqueeze(1),
    )
    scores = F.relu(scores) * self.softmax_scale
    weights = self.weights_proj(hidden_states).float() * self.weights_scaling
    return (scores * weights.unsqueeze(-1)).sum(dim=2)


class DeepSeekV4Indexer(nn.Module):
  """Stateless DeepSeek V4 Lightning Indexer."""

  def __init__(
    self,
    d_model: int,
    q_lora_rank: int,
    compress_rate: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope_base: float = 10_000.0,
    norm_eps: float = 1e-6,
  ) -> None:
    super().__init__()
    if compress_rate <= 0:
      raise ValueError("compress_rate must be positive")
    if index_topk <= 0:
      raise ValueError("index_topk must be positive")

    self.compress_rate = compress_rate
    self.num_heads = index_n_heads
    self.head_dim = index_head_dim
    self.index_topk = index_topk
    self.kv_proj = nn.Linear(d_model, 2 * index_head_dim, bias=False)
    self.gate_proj = nn.Linear(d_model, 2 * index_head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.zeros(compress_rate, 2 * index_head_dim))
    self.kv_norm = RMSNorm(index_head_dim, eps=norm_eps)
    self.q_b_proj = nn.Linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
    self.rotary_emb = DeepSeekV4RotaryEmbedding(
      head_dim=index_head_dim,
      rope_head_dim=index_head_dim,
      base=rope_base,
    )
    self.scorer = DeepSeekV4IndexerScorer(
      d_model=d_model,
      index_n_heads=index_n_heads,
      index_head_dim=index_head_dim,
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    positions: torch.Tensor,
  ) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    compressed_kv = self.compress(hidden_states)

    cos_q, sin_q = self.rotary_emb(hidden_states, positions=positions)
    q = self.q_b_proj(q_residual)
    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    q = apply_deepseek_v4_rotary(q, cos_q, sin_q).transpose(1, 2)

    index_scores = self.scorer(q, compressed_kv, hidden_states)
    compressed_len = compressed_kv.shape[1]
    top_k = min(self.index_topk, compressed_len)

    if compressed_len > 0:
      if positions.ndim == 1:
        positions = positions.unsqueeze(0)
      positions = positions.expand(batch_size, -1)
      causal_threshold = (positions + 1) // self.compress_rate
      entry_indices = torch.arange(compressed_len, device=index_scores.device)
      future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
      index_scores = index_scores.masked_fill(future_mask, float("-inf"))
      top_k_indices = index_scores.topk(top_k, dim=-1).indices
      invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
      return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

    return index_scores.topk(top_k, dim=-1).indices

  def compress(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
    chunk_kv = kv[:, :usable]
    chunk_gate = gate[:, :usable]
    if chunk_kv.shape[1] == 0:
      return chunk_kv.new_zeros((batch_size, 0, self.head_dim))

    n_windows = chunk_kv.shape[1] // self.compress_rate
    ratio = self.compress_rate
    chunk_kv = chunk_kv.view(batch_size, n_windows, ratio, -1)
    chunk_gate = chunk_gate.view(batch_size, n_windows, ratio, -1) + self.position_bias

    new_kv = chunk_kv.new_zeros((batch_size, n_windows, 2 * ratio, self.head_dim))
    new_gate = chunk_gate.new_full(
      (batch_size, n_windows, 2 * ratio, self.head_dim),
      float("-inf"),
    )
    new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
    new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
    if n_windows > 1:
      new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
      new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]

    compressed = self.kv_norm(
      (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(
        dim=2,
      )
    )
    positions = torch.arange(n_windows, device=compressed.device)
    positions = positions * self.compress_rate
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    cos, sin = self.rotary_emb(compressed, positions=positions)
    return apply_deepseek_v4_rotary(compressed.unsqueeze(1), cos, sin).squeeze(1)


class DeepSeekV4HCACompressor(nn.Module):
  """Stateless DeepSeek V4 heavily-compressed attention compressor."""

  def __init__(
    self,
    d_model: int,
    head_dim: int,
    compress_rate: int,
    rope_head_dim: int | None = None,
    rope_base: float = 10_000.0,
    norm_eps: float = 1e-6,
  ) -> None:
    super().__init__()
    if d_model <= 0:
      raise ValueError("d_model must be positive")
    if head_dim <= 0:
      raise ValueError("head_dim must be positive")
    if compress_rate <= 0:
      raise ValueError("compress_rate must be positive")

    rope_head_dim = rope_head_dim if rope_head_dim is not None else head_dim
    self.compress_rate = compress_rate
    self.head_dim = head_dim
    self.kv_proj = nn.Linear(d_model, head_dim, bias=False)
    self.gate_proj = nn.Linear(d_model, head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.zeros(compress_rate, head_dim))
    self.kv_norm = RMSNorm(head_dim, eps=norm_eps)
    self.rotary_emb = DeepSeekV4RotaryEmbedding(
      head_dim=head_dim,
      rope_head_dim=rope_head_dim,
      base=rope_base,
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    compressed = self.compress(hidden_states)
    compressed_kv = compressed.unsqueeze(1)
    compressed_len = compressed_kv.shape[2]
    seq_len = positions.shape[-1]
    if seq_len == 1 or compressed_len == 0:
      return compressed_kv, None

    if positions.ndim == 1:
      positions = positions.unsqueeze(0)
    positions = positions.expand(hidden_states.shape[0], -1)
    entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
    causal_threshold = (positions + 1) // self.compress_rate
    block_bias = compressed_kv.new_zeros(
      (hidden_states.shape[0], 1, seq_len, compressed_len),
    )
    block_bias = block_bias.masked_fill(
      entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
      float("-inf"),
    )
    return compressed_kv, block_bias

  def compress(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
    chunk_kv = kv[:, :usable]
    chunk_gate = gate[:, :usable]
    if chunk_kv.shape[1] == 0:
      return chunk_kv.new_zeros((batch_size, 0, self.head_dim))

    n_windows = chunk_kv.shape[1] // self.compress_rate
    chunk_kv = chunk_kv.view(batch_size, n_windows, self.compress_rate, -1)
    chunk_gate = (
      chunk_gate.view(batch_size, n_windows, self.compress_rate, -1)
      + self.position_bias
    )
    compressed = self.kv_norm(
      (
        chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)
      ).sum(dim=2)
    )
    positions = torch.arange(n_windows, device=compressed.device)
    positions = positions * self.compress_rate
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    cos, sin = self.rotary_emb(compressed, positions=positions)
    return apply_deepseek_v4_rotary(compressed.unsqueeze(1), cos, sin).squeeze(1)


class DeepSeekV4CSACompressor(nn.Module):
  """Stateless DeepSeek V4 compressed sparse attention compressor."""

  def __init__(
    self,
    d_model: int,
    head_dim: int,
    q_lora_rank: int,
    compress_rate: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope_head_dim: int | None = None,
    rope_base: float = 10_000.0,
    norm_eps: float = 1e-6,
  ) -> None:
    super().__init__()
    if d_model <= 0:
      raise ValueError("d_model must be positive")
    if head_dim <= 0:
      raise ValueError("head_dim must be positive")
    if compress_rate <= 0:
      raise ValueError("compress_rate must be positive")

    rope_head_dim = rope_head_dim if rope_head_dim is not None else head_dim
    self.compress_rate = compress_rate
    self.head_dim = head_dim
    self.kv_proj = nn.Linear(d_model, 2 * head_dim, bias=False)
    self.gate_proj = nn.Linear(d_model, 2 * head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.zeros(compress_rate, 2 * head_dim))
    self.kv_norm = RMSNorm(head_dim, eps=norm_eps)
    self.rotary_emb = DeepSeekV4RotaryEmbedding(
      head_dim=head_dim,
      rope_head_dim=rope_head_dim,
      base=rope_base,
    )
    self.indexer = DeepSeekV4Indexer(
      d_model=d_model,
      q_lora_rank=q_lora_rank,
      compress_rate=compress_rate,
      index_n_heads=index_n_heads,
      index_head_dim=index_head_dim,
      index_topk=index_topk,
      rope_base=rope_base,
      norm_eps=norm_eps,
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    positions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = hidden_states.shape
    compressed = self.compress(hidden_states)
    compressed_kv = compressed.unsqueeze(1)
    top_k_indices = self.indexer(hidden_states, q_residual, positions)
    compressed_len = compressed_kv.shape[2]
    valid = top_k_indices >= 0
    safe_indices = torch.where(
      valid,
      top_k_indices,
      torch.full_like(top_k_indices, compressed_len),
    )
    block_bias = compressed_kv.new_full(
      (batch_size, 1, seq_len, compressed_len + 1),
      float("-inf"),
    )
    block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
    return compressed_kv, block_bias[..., :compressed_len]

  def compress(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
    chunk_kv = kv[:, :usable]
    chunk_gate = gate[:, :usable]
    if chunk_kv.shape[1] == 0:
      return chunk_kv.new_zeros((batch_size, 0, self.head_dim))

    n_windows = chunk_kv.shape[1] // self.compress_rate
    ratio = self.compress_rate
    chunk_kv = chunk_kv.view(batch_size, n_windows, ratio, -1)
    chunk_gate = chunk_gate.view(batch_size, n_windows, ratio, -1) + self.position_bias

    new_kv = chunk_kv.new_zeros((batch_size, n_windows, 2 * ratio, self.head_dim))
    new_gate = chunk_gate.new_full(
      (batch_size, n_windows, 2 * ratio, self.head_dim),
      float("-inf"),
    )
    new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
    new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
    if n_windows > 1:
      new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
      new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]

    compressed = self.kv_norm(
      (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(
        dim=2,
      )
    )
    positions = torch.arange(n_windows, device=compressed.device)
    positions = positions * self.compress_rate
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    cos, sin = self.rotary_emb(compressed, positions=positions)
    return apply_deepseek_v4_rotary(compressed.unsqueeze(1), cos, sin).squeeze(1)
