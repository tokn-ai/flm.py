import torch
from flm_modules import (
  DeepSeekV4Attention,
  DeepSeekV4RotaryEmbedding,
  apply_deepseek_v4_rotary,
)
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
  DeepseekV4Attention,
  DeepseekV4RotaryEmbedding,
  apply_rotary_pos_emb,
)


def test_deepseek_v4_rotary_embedding_matches_transformers(random_input) -> None:
  config = _deepseek_v4_config()
  reference = DeepseekV4RotaryEmbedding(config)
  layer = DeepSeekV4RotaryEmbedding(
    head_dim=4,
    rope_head_dim=4,
    base=10_000.0,
  )
  x = random_input(2, 5, 8)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

  actual_cos, actual_sin = layer(x, positions=position_ids)
  expected_cos, expected_sin = reference(x, position_ids, layer_type="main")

  torch.testing.assert_close(actual_cos, expected_cos)
  torch.testing.assert_close(actual_sin, expected_sin)


def test_apply_deepseek_v4_rotary_matches_transformers(random_input) -> None:
  x = random_input(2, 3, 5, 6)
  cos = random_input(2, 5, 2)
  sin = random_input(2, 5, 2)

  torch.testing.assert_close(
    apply_deepseek_v4_rotary(x, cos, sin),
    apply_rotary_pos_emb(x, cos, sin),
  )


def test_deepseek_v4_attention_matches_transformers_sliding_attention(
  random_input,
) -> None:
  config = _deepseek_v4_config()
  reference = DeepseekV4Attention(config, layer_idx=0)
  layer = DeepSeekV4Attention(
    d_model=8,
    n_heads=2,
    head_dim=4,
    q_lora_rank=5,
    o_lora_rank=3,
    o_groups=2,
    rope_head_dim=4,
    bias=False,
    rope_base=10_000.0,
    norm_eps=1e-6,
  )
  x = random_input(2, 5, 8)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
  position_embeddings = {
    "main": DeepseekV4RotaryEmbedding(config)(x, position_ids, layer_type="main")
  }
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)

  with torch.no_grad():
    layer.q_a_proj.weight.copy_(reference.q_a_proj.weight)
    layer.q_a_norm.weight.copy_(reference.q_a_norm.weight)
    layer.q_b_proj.weight.copy_(reference.q_b_proj.weight)
    layer.kv_proj.weight.copy_(reference.kv_proj.weight)
    layer.kv_norm.weight.copy_(reference.kv_norm.weight)
    layer.o_a_proj.weight.copy_(reference.o_a_proj.weight)
    layer.o_b_proj.weight.copy_(reference.o_b_proj.weight)
    layer.sinks.copy_(torch.linspace(-0.2, 0.2, steps=2))
    reference.sinks.copy_(layer.sinks)

  expected, _ = reference(
    x,
    position_embeddings=position_embeddings,
    position_ids=position_ids,
    attention_mask=attention_mask,
  )

  torch.testing.assert_close(
    layer(x, attention_mask=attention_mask, positions=position_ids),
    expected,
  )


def test_deepseek_v4_attention_backpropagates(random_input) -> None:
  layer = DeepSeekV4Attention(
    d_model=8,
    n_heads=2,
    head_dim=4,
    q_lora_rank=5,
    o_lora_rank=3,
    o_groups=2,
    rope_head_dim=4,
  )
  x = random_input(2, 5, 8).requires_grad_()

  layer(x).square().mean().backward()

  assert x.grad is not None
  assert layer.q_a_proj.weight.grad is not None
  assert layer.kv_proj.weight.grad is not None
  assert layer.sinks.grad is not None
  assert layer.o_b_proj.weight.grad is not None


def _deepseek_v4_config() -> DeepseekV4Config:
  return DeepseekV4Config(
    vocab_size=32,
    hidden_size=8,
    num_hidden_layers=1,
    num_attention_heads=2,
    head_dim=4,
    q_lora_rank=5,
    o_lora_rank=3,
    o_groups=2,
    layer_types=["sliding_attention"],
    rope_parameters={
      "main": {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
      },
      "compress": {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
      },
    },
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    sliding_window=5,
    _attn_implementation="eager",
  )


def _causal_mask(batch_size: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
  mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min)
  mask = torch.triu(mask, diagonal=1)
  return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1)
