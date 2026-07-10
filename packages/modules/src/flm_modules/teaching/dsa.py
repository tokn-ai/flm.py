"""Teaching scaffold for DeepSeek sparse attention."""

# ruff: noqa: E501,F722

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn

from flm_modules.attentions.backends import (
  AttentionBackend,
  scaled_dot_product_attention,
)
from flm_modules.norm import LayerNorm
from flm_modules.rope import RopeLayout, RotaryEmbedding, apply_rotary

from .mla import DeepSeekMLA


class DeepSeekDSAIndexer(nn.Module):
  """DeepSeek V3.2 sparse attention indexer scaffold."""

  def __init__(
    self,
    d_model: int,
    q_lora_rank: int,
    qk_rope_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    norm_eps: float = 1e-6,
    rope_base: float = 10_000.0,
    rope_layout: RopeLayout = RopeLayout.LLAMA,
    causal: bool = True,
  ) -> None:
    super().__init__()
    if qk_rope_head_dim > index_head_dim:
      raise ValueError("qk_rope_head_dim must not exceed index_head_dim")
    self.d_model = d_model
    self.q_lora_rank = q_lora_rank
    self.qk_rope_head_dim = qk_rope_head_dim
    self.qk_nope_head_dim = index_head_dim - qk_rope_head_dim
    self.n_heads = index_n_heads
    self.head_dim = index_head_dim
    self.index_topk = index_topk
    self.causal = causal
    self.rope = RotaryEmbedding(qk_rope_head_dim, base=rope_base, layout=rope_layout)

    self.wq_u_proj = nn.Linear(q_lora_rank, self.n_heads * self.head_dim, bias=False)
    self.wk_u_proj = nn.Linear(d_model, self.head_dim, bias=False)
    self.k_norm = LayerNorm(self.head_dim, eps=norm_eps)
    self.weights_proj = nn.Linear(d_model, self.n_heads, bias=False)

    self.scaling = (self.n_heads * self.head_dim) ** -0.5

  def forward(
    self,
    hidden_states: Float[torch.Tensor, "... seq d_model"],
    q_residual: Float[torch.Tensor, "... seq q_lora_rank"],
    attention_mask: Float[torch.Tensor, "... seq seq"] | None = None,
    positions: Float[torch.Tensor, "... seq seq"] | None = None,
  ) -> Int[torch.Tensor, "... seq topk"]:
    q_base: Float[torch.Tensor, "... n_heads seq head_dim"] = rearrange(
      self.wq_u_proj(q_residual),
      "... seq (n_heads head_dim) -> ... n_heads seq head_dim",
      n_heads=self.n_heads,
    )
    q_nope, q_rope = torch.split(
      q_base, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    cos, sin = self.rope(q_rope, positions)
    q_rope: Float[torch.Tensor, "... n_heads seq qk_rope_head_dim"] = apply_rotary(
      q_rope, cos, sin
    )
    q = torch.cat([q_nope, q_rope], dim=-1)
    k_base: Float[torch.Tensor, "... 1 seq head_dim"] = rearrange(
      self.wk_u_proj(hidden_states), "... seq head_dim -> ... 1 seq head_dim"
    )
    k_base = self.k_norm(k_base)
    k_nope, k_rope = torch.split(
      k_base, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    k_rope: Float[torch.Tensor, "... 1 seq qk_rope_head_dim"] = apply_rotary(
      k_rope, cos, sin
    )
    k = torch.cat([k_nope, k_rope], dim=-1)
    weights: Float[torch.Tensor, "... seq n_heads"] = (
      self.weights_proj(hidden_states) * self.scaling
    )

    scores = self._index_scores(
      q, k, weights, attention_mask=attention_mask, causal=self.causal
    )
    topk_scores, topk_indices = torch.topk(scores, self.index_topk, dim=-1)
    return topk_indices.to(torch.int32)

  def _index_scores(
    self,
    q: Float[torch.Tensor, "... n_heads seq head_dim"],
    k: Float[torch.Tensor, "... 1 seq head_dim"],
    weights: Float[torch.Tensor, "... seq n_heads"],
    attention_mask: Float[torch.Tensor, "... seq seq"] | None = None,
    causal: bool = True,
  ) -> Float[torch.Tensor, "... n_heads seq seq"]:
    k = rearrange(k, "... 1 seq head_dim -> ... seq head_dim")
    qk = einsum(
      q, k, "... n_heads seq_q head_dim, ... seq_k head_dim -> ... seq_q seq_k n_heads"
    )
    qk = F.relu(qk)  # relu before mask
    scores = einsum(
      qk, weights, "... seq_q seq_k n_heads, ... seq_q n_heads -> ... seq_q seq_k"
    )
    if attention_mask is not None:
      scores = scores + attention_mask
    elif causal:
      seq_len = q.size(-2)
      attention_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
      scores = scores + attention_mask
    return scores

  def set_weights_from_transformers(
    self,
    wq_b: Tensor,
    wk: Tensor,
    k_norm_weight: Tensor,
    k_norm_bias: Tensor,
    weights_proj: Tensor,
  ) -> None:
    self.wq_u_proj.weight.data.copy_(self._head_weight_from_transformers(wq_b))
    self.wk_u_proj.weight.data.copy_(self._head_vector_from_transformers(wk))
    self.weights_proj.weight.data.copy_(weights_proj)
    self.k_norm.weight.data.copy_(self._head_vector_from_transformers(k_norm_weight))
    self.k_norm.bias.data.copy_(self._head_vector_from_transformers(k_norm_bias))

  def _head_weight_from_transformers(self, weight: Tensor) -> Tensor:
    weight = weight.view(self.n_heads, self.head_dim, self.q_lora_rank)
    rope, nope = weight.split([self.qk_rope_head_dim, self.qk_nope_head_dim], dim=1)
    return torch.cat((nope, rope), dim=1).reshape(
      self.n_heads * self.head_dim,
      self.q_lora_rank,
    )

  def _head_vector_from_transformers(self, tensor: Tensor) -> Tensor:
    rope, nope = tensor.split([self.qk_rope_head_dim, self.qk_nope_head_dim], dim=0)
    return torch.cat((nope, rope), dim=0)


class DeepSeekDSA(nn.Module):
  """DeepSeek V3.2 sparse attention scaffold."""

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
    rope_layout: RopeLayout = RopeLayout.DEEPSEEK_V32,
    index_rope_layout: RopeLayout = RopeLayout.LLAMA,
    norm_eps: float = 1e-6,
    causal: bool = True,
    backend: AttentionBackend | str = AttentionBackend.TORCH,
  ) -> None:
    super().__init__()
    self.mla = DeepSeekMLA(
      d_model=d_model,
      n_heads=n_heads,
      kv_lora_rank=kv_lora_rank,
      q_lora_rank=q_lora_rank,
      qk_nope_head_dim=qk_nope_head_dim,
      qk_rope_head_dim=qk_rope_head_dim,
      v_head_dim=v_head_dim,
      bias=bias,
      rope_base=rope_base,
      rope_layout=rope_layout,
      norm_eps=norm_eps,
      causal=causal,
      backend=backend,
    )
    self.indexer = DeepSeekDSAIndexer(
      d_model=d_model,
      q_lora_rank=q_lora_rank,
      qk_rope_head_dim=qk_rope_head_dim,
      index_n_heads=index_n_heads,
      index_head_dim=index_head_dim,
      index_topk=index_topk,
      rope_base=rope_base,
      rope_layout=index_rope_layout,
      causal=causal,
    )

  def forward(
    self,
    hidden_states: Float[torch.Tensor, "... seq d_model"],
    attention_mask: Float[torch.Tensor, "... 1 seq seq"] | None = None,
    positions: Float[torch.Tensor, "... seq"] | None = None,
  ) -> Float[torch.Tensor, "... seq d_model"]:
    # although we should first select c_kv then do projection,
    # it is mathematically equivalent, we just do a demo,
    # and never use this in production, because it is not efficient.
    c_kv, c_q = self.mla._compress(hidden_states)
    q, k, v = self.mla._project_qkv(hidden_states, c_kv, c_q, positions=positions)

    indexer_mask = attention_mask[..., 0, :, :] if attention_mask is not None else None
    indices: Int[torch.Tensor, "... seq topk"] = self.indexer.forward(
      hidden_states, c_q, attention_mask=indexer_mask, positions=positions
    )
    # we should create a sparse_mask here, since k, v is [BHSD], after slicing, it has to be BHQKD (note Q and K here)
    # since different query would have different keys set (according to indcies), it would not fit in scaled_dot_product_attention
    # so the selecting isn't train low cost, it only saves cost when inference.
    sparse_mask = indices.new_ones((*indices.shape[:-1], k.shape[-2]), dtype=torch.bool)
    sparse_mask = sparse_mask.scatter(-1, indices.long(), False).unsqueeze(-3)
    attention_mask = (
      hidden_states.new_zeros(
        (*sparse_mask.shape[:-3], 1, sparse_mask.shape[-2], sparse_mask.shape[-1])
      )
      if attention_mask is None
      else attention_mask
    )
    attention_mask = attention_mask.masked_fill(
      sparse_mask, torch.finfo(hidden_states.dtype).min
    )
    o = scaled_dot_product_attention(
      q,
      k,
      v,
      backend=self.mla.backend,
      attn_mask=attention_mask,
      causal=self.mla.causal,
      scale=self.mla.scaling,
    )
    return self.mla._project_o(o)

  def set_weights_from_transformers(
    self,
    q_a_proj: Tensor,
    q_a_layernorm: Tensor,
    q_b_proj: Tensor,
    kv_a_proj_with_mqa: Tensor,
    kv_a_layernorm: Tensor,
    kv_b_proj: Tensor,
    o_proj: Tensor,
    indexer_wq_b: Tensor,
    indexer_wk: Tensor,
    indexer_k_norm_weight: Tensor,
    indexer_k_norm_bias: Tensor,
    indexer_weights_proj: Tensor,
  ) -> None:
    self.mla.set_weights_from_transformers(
      q_proj=None,
      q_a_proj=q_a_proj,
      q_a_layernorm=q_a_layernorm,
      q_b_proj=q_b_proj,
      kv_a_proj_with_mqa=kv_a_proj_with_mqa,
      kv_a_layernorm=kv_a_layernorm,
      kv_b_proj=kv_b_proj,
      o_proj=o_proj,
    )
    self.indexer.set_weights_from_transformers(
      wq_b=indexer_wq_b,
      wk=indexer_wk,
      k_norm_weight=indexer_k_norm_weight,
      k_norm_bias=indexer_k_norm_bias,
      weights_proj=indexer_weights_proj,
    )
