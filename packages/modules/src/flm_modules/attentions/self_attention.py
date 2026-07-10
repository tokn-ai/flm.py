"""Self-attention layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from flm_modules.attentions.backends import (
  AttentionBackend,
  scaled_dot_product_attention,
)
from flm_modules.rope import RotaryEmbedding, apply_rotary


class SelfAttention(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_heads: int,
    bias: bool = False,
    rope_base: float = 10_000.0,
    backend: AttentionBackend | str = AttentionBackend.TORCH,
    causal: bool = True,
  ) -> None:
    super().__init__()
    if d_model % n_heads != 0:
      raise ValueError("d_model must be divisible by n_heads")

    self.d_model = d_model
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.backend = AttentionBackend(backend)
    self.causal = causal

    self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
    self.out = nn.Linear(d_model, d_model, bias=bias)
    self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

  def forward(
    self,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rope(q)
    q = apply_rotary(q, cos, sin, layout=self.rope.layout)
    k = apply_rotary(k, cos, sin, layout=self.rope.layout)
    y = scaled_dot_product_attention(
      q,
      k,
      v,
      backend=self.backend,
      attn_mask=attn_mask,
      causal=self.causal,
    )
    y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    return self.out(y)


class QKNormSelfAttention(nn.Module):
  """RoPE attention with per-head QK normalization and value residuals.

  ``value_residual`` is expected in head-major ``[B, H, T, D]`` layout. When
  supplied, ``value_mix`` interpolates between the current values and that
  residual. The method also returns the pre-mix values so the first attention
  layer can seed a value-residual stream.
  """

  def __init__(
    self,
    d_model: int,
    n_heads: int,
    bias: bool = False,
    rope_base: float = 10_000.0,
    norm_eps: float = 1e-6,
    backend: AttentionBackend | str = AttentionBackend.TORCH,
    causal: bool = True,
    *,
    zero_init_out: bool = False,
    paired_heads: bool = False,
  ) -> None:
    super().__init__()
    if d_model % n_heads != 0:
      raise ValueError("d_model must be divisible by n_heads")
    self.d_model = d_model
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.norm_eps = norm_eps
    self.backend = AttentionBackend(backend)
    self.causal = causal
    self.paired_heads = paired_heads
    if paired_heads and n_heads % 2:
      raise ValueError("paired-head attention requires an even number of heads")
    self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
    self.out = nn.Linear(d_model, d_model, bias=bias)
    rope_dim = 2 * self.head_dim if paired_heads else self.head_dim
    self.rope = RotaryEmbedding(rope_dim, base=rope_base)
    if zero_init_out:
      nn.init.zeros_(self.out.weight)
      if self.out.bias is not None:
        nn.init.zeros_(self.out.bias)

  def forward(
    self,
    x: torch.Tensor,
    *,
    value_residual: torch.Tensor | None = None,
    value_mix: torch.Tensor | float | None = None,
    auxiliary_values: torch.Tensor | None = None,
    partial_key_offset: bool = False,
    output_gate_weight: torch.Tensor | None = None,
    xsa_alpha: torch.Tensor | None = None,
    attention_window: int | None = None,
    attn_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q = self._split_heads(q)
    k = self._split_heads(k)
    v = self._split_heads(v)

    q = self._rms_norm(q)
    k = self._rms_norm(k)
    current_values = v
    if value_residual is not None:
      if value_residual.shape != v.shape:
        raise ValueError("value_residual must have the same shape as values")
      mix = 0.5 if value_mix is None else value_mix
      v = torch.lerp(value_residual, v, mix)
    if auxiliary_values is not None:
      if auxiliary_values.shape != v.shape:
        raise ValueError("auxiliary_values must have the same shape as values")
      v = v + auxiliary_values

    if self.paired_heads:
      if partial_key_offset:
        raise ValueError("partial key offset is unsupported for paired heads")
      if attn_mask is not None:
        raise ValueError("custom masks are unsupported for paired heads")
      if xsa_alpha is not None:
        raise ValueError("XSA is unsupported for paired heads")
      q = self._pair_qk(q, batch_size=batch_size, seq_len=seq_len)
      k = self._pair_qk(k, batch_size=batch_size, seq_len=seq_len)
      v = self._pair_values(v, batch_size=batch_size, seq_len=seq_len)
    else:
      cos, sin = self.rope(q)
      q = apply_rotary(q, cos, sin, layout=self.rope.layout)
      k = apply_rotary(k, cos, sin, layout=self.rope.layout)
      if partial_key_offset and seq_len > 1:
        k = k.clone()
        k[:, :, 1:, self.head_dim // 2 :] = k[:, :, :-1, self.head_dim // 2 :]

    if attention_window is not None:
      if attention_window < 1:
        raise ValueError("attention_window must be positive")
      if attn_mask is not None:
        raise ValueError("attention_window cannot be combined with attn_mask")
      if not self.causal:
        raise ValueError("windowed QK attention currently requires causal attention")
      attention_length = q.shape[-2]
      positions = torch.arange(attention_length, device=q.device)
      distance = positions[:, None] - positions[None, :]
      attn_mask = (distance >= 0) & (distance <= attention_window)

    y = scaled_dot_product_attention(
      q,
      k,
      v,
      backend=self.backend,
      attn_mask=attn_mask,
      causal=self.causal,
    )
    if self.paired_heads:
      y = self._unpair_output(y, batch_size=batch_size, seq_len=seq_len)
    if xsa_alpha is not None:
      if xsa_alpha.shape != (self.n_heads,):
        raise ValueError("xsa_alpha must have shape [n_heads]")
      normalized_values = F.normalize(v, dim=-1, eps=1e-4)
      projection = (y * normalized_values).sum(dim=-1, keepdim=True)
      alpha = xsa_alpha.tanh().view(1, self.n_heads, 1, 1).to(y.dtype)
      y = y - alpha * projection * normalized_values
    if output_gate_weight is not None:
      if output_gate_weight.ndim != 2 or output_gate_weight.shape[0] != self.n_heads:
        raise ValueError("output_gate_weight must have shape [n_heads, gate_dim]")
      gate_dim = output_gate_weight.shape[1]
      if gate_dim > x.shape[-1]:
        raise ValueError("attention output gate exceeds model dimension")
      gate = torch.sigmoid(F.linear(x[..., :gate_dim], output_gate_weight))
      y = y * gate.transpose(1, 2).unsqueeze(-1)
    y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    return self.out(y), current_values

  def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

  def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return F.rms_norm(
      x.float(),
      (self.head_dim,),
      eps=self.norm_eps,
    ).to(dtype)

  def _pair_qk(
    self,
    x: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
  ) -> torch.Tensor:
    paired = (
      x.transpose(1, 2)
      .contiguous()
      .view(batch_size, seq_len, self.n_heads // 2, 2 * self.head_dim)
      .transpose(1, 2)
    )
    cos, sin = self.rope(paired)
    paired = apply_rotary(paired, cos, sin, layout=self.rope.layout)
    return (
      paired.transpose(1, 2)
      .contiguous()
      .view(batch_size, 2 * seq_len, self.n_heads // 2, self.head_dim)
      .transpose(1, 2)
    )

  def _pair_values(
    self,
    values: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
  ) -> torch.Tensor:
    return (
      values.transpose(1, 2)
      .contiguous()
      .view(batch_size, 2 * seq_len, self.n_heads // 2, self.head_dim)
      .transpose(1, 2)
    )

  def _unpair_output(
    self,
    output: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
  ) -> torch.Tensor:
    return (
      output.transpose(1, 2)
      .contiguous()
      .view(batch_size, seq_len, self.n_heads, self.head_dim)
      .transpose(1, 2)
    )
