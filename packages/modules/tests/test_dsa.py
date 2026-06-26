import pytest
import torch
from flm_modules import DeepSeekDSA, DeepSeekDSAIndexer
from transformers import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
  DeepseekV32Attention,
  DeepseekV32Indexer,
  DeepseekV32RotaryEmbedding,
)


def test_deepseek_dsa_indexer_matches_transformers(random_input) -> None:
  config = _deepseek_v32_config()
  reference = DeepseekV32Indexer(config, layer_idx=0)
  layer = DeepSeekDSAIndexer(
    d_model=8,
    q_lora_rank=5,
    qk_rope_head_dim=4,
    index_n_heads=3,
    index_head_dim=6,
    index_topk=3,
    rope_base=10_000.0,
  )
  x = random_input(2, 5, 8)
  q_residual = random_input(2, 5, 5)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
  position_embeddings = DeepseekV32RotaryEmbedding(config)(x, position_ids)
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)[:, 0]

  with torch.no_grad():
    layer.wq_b.weight.copy_(reference.wq_b.weight)
    layer.wk.weight.copy_(reference.wk.weight)
    layer.k_norm.weight.copy_(reference.k_norm.weight)
    layer.k_norm.bias.copy_(reference.k_norm.bias)
    layer.weights_proj.weight.copy_(reference.weights_proj.weight)

  expected = reference(
    x,
    q_residual,
    position_embeddings,
    attention_mask,
    position_ids,
  )

  torch.testing.assert_close(
    layer(x, q_residual, attention_mask=attention_mask, positions=position_ids),
    expected,
  )


def test_deepseek_dsa_matches_transformers_attention(random_input) -> None:
  config = _deepseek_v32_config()
  reference = DeepseekV32Attention(config, layer_idx=0)
  layer = DeepSeekDSA(
    d_model=8,
    n_heads=2,
    kv_lora_rank=6,
    q_lora_rank=5,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
    index_n_heads=3,
    index_head_dim=6,
    index_topk=3,
    bias=False,
    rope_base=10_000.0,
    norm_eps=1e-6,
  )
  x = random_input(2, 5, 8)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
  position_embeddings = DeepseekV32RotaryEmbedding(config)(x, position_ids)
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)

  with torch.no_grad():
    layer.q_a_proj.weight.copy_(reference.q_a_proj.weight)
    layer.q_a_layernorm.weight.copy_(reference.q_a_layernorm.weight)
    layer.q_b_proj.weight.copy_(reference.q_b_proj.weight)
    layer.kv_a_proj_with_mqa.weight.copy_(reference.kv_a_proj_with_mqa.weight)
    layer.kv_a_layernorm.weight.copy_(reference.kv_a_layernorm.weight)
    layer.kv_b_proj.weight.copy_(reference.kv_b_proj.weight)
    layer.o_proj.weight.copy_(reference.o_proj.weight)
    layer.indexer.wq_b.weight.copy_(reference.indexer.wq_b.weight)
    layer.indexer.wk.weight.copy_(reference.indexer.wk.weight)
    layer.indexer.k_norm.weight.copy_(reference.indexer.k_norm.weight)
    layer.indexer.k_norm.bias.copy_(reference.indexer.k_norm.bias)
    layer.indexer.weights_proj.weight.copy_(reference.indexer.weights_proj.weight)

  expected, _ = reference(
    x,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
    position_ids=position_ids,
  )

  torch.testing.assert_close(
    layer(x, attention_mask=attention_mask, positions=position_ids),
    expected,
  )


def test_deepseek_dsa_backpropagates(random_input) -> None:
  layer = DeepSeekDSA(
    d_model=8,
    n_heads=2,
    kv_lora_rank=6,
    q_lora_rank=5,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
    index_n_heads=3,
    index_head_dim=6,
    index_topk=3,
  )
  x = random_input(2, 5, 8).requires_grad_()

  layer(x).square().mean().backward()

  assert x.grad is not None
  assert layer.q_a_proj.weight.grad is not None
  assert layer.kv_a_proj_with_mqa.weight.grad is not None
  assert layer.o_proj.weight.grad is not None


def test_deepseek_dsa_rejects_invalid_index_rope_dimension() -> None:
  with pytest.raises(ValueError, match="must not exceed index_head_dim"):
    DeepSeekDSAIndexer(
      d_model=8,
      q_lora_rank=5,
      qk_rope_head_dim=8,
      index_n_heads=3,
      index_head_dim=6,
      index_topk=3,
    )


def _deepseek_v32_config() -> DeepseekV32Config:
  config = DeepseekV32Config(
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=2,
    q_lora_rank=5,
    kv_lora_rank=6,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
    index_n_heads=3,
    index_head_dim=6,
    index_topk=3,
    attention_bias=False,
    attention_dropout=0.0,
    rope_parameters={
      "rope_type": "default",
      "rope_theta": 10_000.0,
    },
  )
  config._attn_implementation = "eager"
  return config


def _causal_mask(batch_size: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
  mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min)
  mask = torch.triu(mask, diagonal=1)
  return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1)
