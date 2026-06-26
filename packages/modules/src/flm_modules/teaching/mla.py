"""Multi-head latent attention layers."""

# ruff: noqa: E501,F722

from __future__ import annotations

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from flm_modules.attentions.backends import (
  AttentionBackend,
  scaled_dot_product_attention,
)
from flm_modules.norm import RMSNorm
from flm_modules.rope import RotaryEmbedding, apply_rotary

# class DeepSeekLoraRoPE(nn.Module):
#   def __init__(
#     self,
#     d_input: int,
#     d_lora_rank: int,
#     d_nope_head: int,
#     d_rope_head: int,
#     n_heads: int,
#     bias: bool = False,
#     rope_base: float = 10_000.0,
#     rope_layout: str = "llama",
#     norm_eps: float = 1e-6,
#     shared_rope_head: bool = False,
#   ):
#     super().__init__()
#     self.d_input = d_input
#     self.d_lora_rank = d_lora_rank
#     self.d_nope_head = d_nope_head
#     self.d_rope_head = d_rope_head
#     self.n_heads = n_heads
#     self.shared_rope_head = shared_rope_head

#     self.d_head = d_nope_head + d_rope_head

#     self.d_proj = nn.Linear(d_input, d_lora_rank, bias=bias)
#     self.layernorm = RMSNorm(d_lora_rank, eps=norm_eps)
#     self.u_proj = nn.Linear(d_lora_rank, n_heads * d_nope_head, bias=bias)
#     if shared_rope_head:
#       self.r_proj = nn.Linear(d_lora_rank, d_rope_head, bias=bias)
#     else:
#       self.r_proj = nn.Linear(d_lora_rank, n_heads * d_rope_head, bias=bias)

#     self.rope = RotaryEmbedding(
#       d_rope_head, base=rope_base, layout=rope_layout
#     )

#   def forward(
#     self,
#     hidden_states: Float[Tensor, "... seq d_input"],
#     positions: Float[Tensor, "... seq"] | None = None,
#   ) -> tuple[
#     Float[Tensor, "... n_heads seq d_head"],
#     Float[Tensor, "... n_heads seq d_head"],
#   ]:
#     c: Float[Tensor, "... n_heads seq d_lora_rank"] = self._compress(hidden_states)

#   def _compress(
#     self,
#     hidden_states: Float[Tensor, "... seq d_input"],
#   ) -> Float[Tensor, "... n_heads seq d_lora_rank"]:
#     c: Float[Tensor, "... seq d_lora_rank"] = self.layernorm(self.d_proj(hidden_states))
#     return rearrange(c, "... seq (n_heads d_lora_rank) -> ... n_heads seq d_lora_rank", n_heads=self.n_heads)

class DeepSeekMLA(nn.Module):
  """DeepSeek-style multi-head latent attention."""

  def __init__(
    self,
    d_model: int,
    n_heads: int,
    kv_lora_rank: int,  # d_c
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

    self.kv_d_proj = nn.Linear(d_model, self.kv_lora_rank, bias=bias)
    self.kv_d_layernorm = RMSNorm(self.kv_lora_rank, eps=norm_eps)
    if self.q_lora_rank is None:
      self.q_proj = nn.Linear(d_model, n_heads * self.qk_head_dim, bias=bias)
    else:
      self.q_d_proj = nn.Linear(d_model, self.q_lora_rank, bias=bias)
      self.q_d_layernorm = RMSNorm(self.q_lora_rank, eps=norm_eps)
      self.q_u_proj = nn.Linear(
        self.q_lora_rank, n_heads * self.qk_nope_head_dim, bias=bias
      )
      self.q_r_proj = nn.Linear(
        self.q_lora_rank, n_heads * self.qk_rope_head_dim, bias=bias
      )
    self.k_u_proj = nn.Linear(
      self.kv_lora_rank, n_heads * self.qk_nope_head_dim, bias=bias
    )
    self.v_u_proj = nn.Linear(
      self.kv_lora_rank, n_heads * self.v_head_dim, bias=bias
    )

    self.k_r_proj = nn.Linear(
      d_model, self.qk_rope_head_dim, bias=bias
    )
    self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=bias)
    self.rope = RotaryEmbedding(
      self.qk_rope_head_dim, base=rope_base, layout=rope_layout
    )

  def forward(
    self,
    hidden_states: Float[Tensor, "... seq d_model"],
    attention_mask: Float[Tensor, "... seq seq"] | None = None,
    positions: Float[Tensor, "... seq"] | None = None,
  ) -> Float[Tensor, "... seq d_model"]:
    c_kv, c_q = self._compress(hidden_states)
    q, k, v = self._project_qkv(hidden_states, c_kv, c_q, positions=positions)
    o = scaled_dot_product_attention(
      q,
      k,
      v,
      backend=self.backend,
      attn_mask=attention_mask,
      causal=self.causal,
      scale=self.scaling,
    )
    return self._project_o(o)

  def _compress(
      self,
      hidden_states: Float[Tensor, "... seq d_model"]
  ) -> tuple[
    Float[Tensor, "... seq kv_lora_rank"],
    Float[Tensor, "... seq q"],
  ]:
    c_kv: Float[Tensor, "... seq kv_lora_rank"] = self.kv_d_layernorm(
      self.kv_d_proj(hidden_states)
    )
    if self.q_lora_rank is not None:
      c_q: Float[Tensor, "... seq q_lora_rank"] = self.q_d_layernorm(self.q_d_proj(hidden_states))
    else:
      c_q: Float[Tensor, "... seq q"] = self.q_proj(hidden_states)
    return c_kv, c_q

  def _project_qkv(
    self,
    hidden_states: Float[Tensor, "... seq d_model"],
    c_kv: Float[Tensor, "... seq kv_lora_rank"],
    c_q: Float[Tensor, "... seq q_lora_rank"],
    positions: Float[Tensor, "... seq"] | None = None,
  ) -> tuple[
    Float[Tensor, "... n_heads seq qk_head_dim"],
    Float[Tensor, "... n_heads seq qk_head_dim"],
    Float[Tensor, "... n_heads seq v_head_dim"],
  ]:
    k_C: Float[Tensor, "... n_heads seq qk_nope_head_dim"] = rearrange(self.k_u_proj(c_kv), "... seq (n_heads qk_nope_head_dim) -> ... n_heads seq qk_nope_head_dim", n_heads=self.n_heads)
    k_R: Float[Tensor, "... n_heads seq qk_rope_head_dim"] = repeat(self.k_r_proj(hidden_states), "... seq qk_rope_head_dim -> ... n_heads seq qk_rope_head_dim", n_heads=self.n_heads)
    v_C: Float[Tensor, "... n_heads seq v_head_dim"] = rearrange(self.v_u_proj(c_kv), "... seq (n_heads v_head_dim) -> ... n_heads seq v_head_dim", n_heads=self.n_heads)
    if self.q_lora_rank is None:
      q_C, q_R = torch.split(
        c_q,
        [
          self.n_heads * self.qk_nope_head_dim,
          self.n_heads * self.qk_rope_head_dim,
        ],
        dim=-1,
      )
      q_C = rearrange(q_C, "... seq (n_heads qk_nope_head_dim) -> ... n_heads seq qk_nope_head_dim", n_heads=self.n_heads)
      q_R = rearrange(q_R, "... seq (n_heads qk_rope_head_dim) -> ... n_heads seq qk_rope_head_dim", n_heads=self.n_heads)
    else:
      q_C = rearrange(self.q_u_proj(c_q), "... seq (n_heads qk_nope_head_dim) -> ... n_heads seq qk_nope_head_dim", n_heads=self.n_heads)
      q_R = rearrange(self.q_r_proj(c_q), "... seq (n_heads qk_rope_head_dim) -> ... n_heads seq qk_rope_head_dim", n_heads=self.n_heads)

    cos, sin = self.rope(q_C, positions=positions)
    k_R = apply_rotary(k_R, cos, sin, layout=self.rope.layout)
    q_R = apply_rotary(q_R, cos, sin, layout=self.rope.layout)

    k = torch.cat((k_C, k_R), dim=-1)
    q = torch.cat((q_C, q_R), dim=-1)

    return q, k, v_C


  def _project_o(
    self,
    o: Float[Tensor, "... n_heads seq v_head_dim"],
  ) -> Float[Tensor, "... seq d_model"]:
    o = rearrange(o, "... n_heads seq v_head_dim -> ... seq (n_heads v_head_dim)")
    return self.o_proj(o)


  def set_weights_from_transformers(
    self,
    q_a_proj: Tensor | None,
    q_a_layernorm: Tensor | None,
    q_b_proj: Tensor | None,
    q_proj: Tensor | None,
    kv_a_proj_with_mqa: Tensor,
    kv_a_layernorm: Tensor,
    kv_b_proj: Tensor,
    o_proj: Tensor
  ) -> None:
    if q_proj is not None:
      q_proj = q_proj.view(self.n_heads, self.qk_head_dim, self.d_model)
      q_nope, q_rope = q_proj.split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim],
        dim=1,
      )
      self.q_proj.weight.data.copy_(
        torch.cat(
          (
            q_nope.reshape(self.n_heads * self.qk_nope_head_dim, self.d_model),
            q_rope.reshape(self.n_heads * self.qk_rope_head_dim, self.d_model),
          ),
          dim=0,
        ),
      )
    else:
      if q_a_proj is None or q_a_layernorm is None or q_b_proj is None:
        raise ValueError(
          "q_proj is None, but one of q_a_proj, q_a_layernorm, "
          "or q_b_proj is also None"
        )
      self.q_d_proj.weight.data.copy_(q_a_proj)
      self.q_d_layernorm.weight.data.copy_(q_a_layernorm)
      q_b_proj = q_b_proj.view(self.n_heads, self.qk_head_dim, self.q_lora_rank)
      q_nope, q_rope = q_b_proj.split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim],
        dim=1,
      )
      self.q_u_proj.weight.data.copy_(
        q_nope.reshape(self.n_heads * self.qk_nope_head_dim, self.q_lora_rank)
      )
      self.q_r_proj.weight.data.copy_(
        q_rope.reshape(self.n_heads * self.qk_rope_head_dim, self.q_lora_rank)
      )
    self.kv_d_proj.weight.data.copy_(
      kv_a_proj_with_mqa[..., :self.kv_lora_rank, :]
    )
    self.kv_d_layernorm.weight.data.copy_(kv_a_layernorm)
    kv_b_proj = kv_b_proj.view(
      self.n_heads,
      self.qk_nope_head_dim + self.v_head_dim,
      self.kv_lora_rank,
    )
    k_nope, v = kv_b_proj.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
    self.k_u_proj.weight.data.copy_(
      k_nope.reshape(self.n_heads * self.qk_nope_head_dim, self.kv_lora_rank)
    )
    self.v_u_proj.weight.data.copy_(
      v.reshape(self.n_heads * self.v_head_dim, self.kv_lora_rank)
    )
    self.k_r_proj.weight.data.copy_(
      kv_a_proj_with_mqa[..., self.kv_lora_rank:, :]
    )
    self.o_proj.weight.data.copy_(o_proj)
