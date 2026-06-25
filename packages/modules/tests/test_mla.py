import pytest
import torch
from flm_modules import DeepSeekMLA
from torch.nn import functional as F
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
  DeepseekV3Attention,
  DeepseekV3RotaryEmbedding,
)


def test_deepseek_mla_preserves_model_dimension(random_input) -> None:
  layer = DeepSeekMLA(
    d_model=8,
    n_heads=2,
    kv_lora_rank=6,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
  )
  x = random_input(2, 5, 8)

  y = layer(x)

  assert y.shape == x.shape


def test_deepseek_mla_supports_q_lora_variant(random_input) -> None:
  layer = DeepSeekMLA(
    d_model=8,
    n_heads=2,
    kv_lora_rank=6,
    q_lora_rank=5,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
  )
  x = random_input(2, 5, 8)

  y = layer(x)

  assert y.shape == x.shape


def test_deepseek_mla_matches_manual_computation(random_input) -> None:
  layer = DeepSeekMLA(
    d_model=8,
    n_heads=2,
    kv_lora_rank=6,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
  )
  x = random_input(2, 5, 8)
  batch_size, seq_len, _ = x.shape

  q = layer.q_proj(x).view(batch_size, seq_len, 2, 8).transpose(1, 2)
  q_pass, q_rot = q.split([4, 4], dim=-1)
  compressed_kv = layer.kv_a_proj_with_mqa(x)
  k_latent, k_rot = compressed_kv.split([6, 4], dim=-1)
  kv = layer.kv_b_proj(layer.kv_a_layernorm(k_latent))
  kv = kv.view(batch_size, seq_len, 2, 8).transpose(1, 2)
  k_pass, v = kv.split([4, 4], dim=-1)
  k_rot = k_rot.view(batch_size, 1, seq_len, 4)
  q_rot, k_rot = layer.rope(q_rot, k_rot)
  q = torch.cat((q_pass, q_rot), dim=-1)
  k = torch.cat((k_pass, k_rot.expand(*k_pass.shape[:-1], -1)), dim=-1)
  expected = F.scaled_dot_product_attention(
    q,
    k,
    v,
    dropout_p=0.0,
    is_causal=True,
    scale=layer.scaling,
  )
  expected = expected.transpose(1, 2).contiguous().view(batch_size, seq_len, 8)
  expected = layer.o_proj(expected)

  torch.testing.assert_close(layer(x), expected)


def test_deepseek_mla_matches_transformers_attention(random_input) -> None:
  config = _deepseek_config(q_lora_rank=None)
  reference = DeepseekV3Attention(config, layer_idx=0)
  layer = DeepSeekMLA(
    d_model=8,
    n_heads=2,
    kv_lora_rank=6,
    q_lora_rank=None,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
    bias=False,
    rope_layout="llama",
  )
  x = random_input(2, 5, 8)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
  rope = DeepseekV3RotaryEmbedding(config)
  position_embeddings = rope(x, position_ids)
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)

  with torch.no_grad():
    layer.q_proj.weight.copy_(reference.q_proj.weight)
    layer.kv_a_proj_with_mqa.weight.copy_(reference.kv_a_proj_with_mqa.weight)
    layer.kv_a_layernorm.weight.copy_(reference.kv_a_layernorm.weight)
    layer.kv_b_proj.weight.copy_(reference.kv_b_proj.weight)
    layer.o_proj.weight.copy_(reference.o_proj.weight)

  expected, _ = reference(
    x,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
  )

  torch.testing.assert_close(layer(x, attention_mask=attention_mask), expected)


def test_deepseek_mla_q_lora_matches_transformers_attention(random_input) -> None:
  config = _deepseek_config(q_lora_rank=5)
  reference = DeepseekV3Attention(config, layer_idx=0)
  layer = DeepSeekMLA(
    d_model=8,
    n_heads=2,
    kv_lora_rank=6,
    q_lora_rank=5,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
    bias=False,
    rope_layout="llama",
  )
  x = random_input(2, 5, 8)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
  rope = DeepseekV3RotaryEmbedding(config)
  position_embeddings = rope(x, position_ids)
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)

  with torch.no_grad():
    layer.q_a_proj.weight.copy_(reference.q_a_proj.weight)
    layer.q_a_layernorm.weight.copy_(reference.q_a_layernorm.weight)
    layer.q_b_proj.weight.copy_(reference.q_b_proj.weight)
    layer.kv_a_proj_with_mqa.weight.copy_(reference.kv_a_proj_with_mqa.weight)
    layer.kv_a_layernorm.weight.copy_(reference.kv_a_layernorm.weight)
    layer.kv_b_proj.weight.copy_(reference.kv_b_proj.weight)
    layer.o_proj.weight.copy_(reference.o_proj.weight)

  expected, _ = reference(
    x,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
  )

  torch.testing.assert_close(layer(x, attention_mask=attention_mask), expected)


def test_deepseek_mla_backpropagates(random_input) -> None:
  layer = DeepSeekMLA(
    d_model=8,
    n_heads=2,
    kv_lora_rank=6,
    q_lora_rank=5,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
  )
  x = random_input(2, 5, 8).requires_grad_()

  layer(x).square().mean().backward()

  assert x.grad is not None
  assert layer.q_a_proj.weight.grad is not None
  assert layer.kv_a_proj_with_mqa.weight.grad is not None
  assert layer.o_proj.weight.grad is not None


def test_deepseek_mla_rejects_invalid_rope_dimension() -> None:
  with pytest.raises(ValueError, match="positive even"):
    DeepSeekMLA(
      d_model=8,
      n_heads=2,
      kv_lora_rank=6,
      qk_nope_head_dim=4,
      qk_rope_head_dim=3,
      v_head_dim=4,
    )


def _deepseek_config(q_lora_rank: int | None) -> DeepseekV3Config:
  return DeepseekV3Config(
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=2,
    q_lora_rank=q_lora_rank,
    kv_lora_rank=6,
    qk_rope_head_dim=4,
    qk_nope_head_dim=4,
    qk_head_dim=8,
    v_head_dim=4,
    rope_interleave=False,
    attention_bias=False,
    attention_dropout=0.0,
  )


def _causal_mask(batch_size: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
  mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min)
  mask = torch.triu(mask, diagonal=1)
  return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1)
