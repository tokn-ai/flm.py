import torch
from flm_modules import (
  DeepSeekV4Attention,
  DeepSeekV4CSACompressor,
  DeepSeekV4HCACompressor,
  DeepSeekV4Indexer,
  DeepSeekV4IndexerScorer,
  DeepSeekV4RotaryEmbedding,
  apply_deepseek_v4_rotary,
)
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
  DeepseekV4Attention,
  DeepseekV4CSACompressor,
  DeepseekV4HCACompressor,
  DeepseekV4Indexer,
  DeepseekV4IndexerScorer,
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


def test_deepseek_v4_indexer_scorer_matches_transformers(random_input) -> None:
  config = _deepseek_v4_config()
  reference = DeepseekV4IndexerScorer(config)
  layer = DeepSeekV4IndexerScorer(
    d_model=8,
    index_n_heads=2,
    index_head_dim=4,
  )
  q = random_input(2, 5, 2, 4)
  compressed_kv = random_input(2, 3, 4)
  hidden_states = random_input(2, 5, 8)

  with torch.no_grad():
    layer.weights_proj.weight.copy_(reference.weights_proj.weight)

  torch.testing.assert_close(
    layer(q, compressed_kv, hidden_states),
    reference(q, compressed_kv, hidden_states),
  )


def test_deepseek_v4_indexer_matches_transformers(random_input) -> None:
  config = _deepseek_v4_config()
  reference = DeepseekV4Indexer(config)
  layer = DeepSeekV4Indexer(
    d_model=8,
    q_lora_rank=5,
    compress_rate=2,
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    rope_base=10_000.0,
    norm_eps=1e-6,
  )
  hidden_states = random_input(2, 6, 8)
  q_residual = random_input(2, 6, 5)
  position_ids = torch.arange(6).unsqueeze(0).expand(2, -1)

  with torch.no_grad():
    reference.position_bias.copy_(
      torch.linspace(-0.2, 0.2, steps=reference.position_bias.numel()).view_as(
        reference.position_bias,
      )
    )
    layer.kv_proj.weight.copy_(reference.kv_proj.weight)
    layer.gate_proj.weight.copy_(reference.gate_proj.weight)
    layer.position_bias.copy_(reference.position_bias)
    layer.kv_norm.weight.copy_(reference.kv_norm.weight)
    layer.q_b_proj.weight.copy_(reference.q_b_proj.weight)
    layer.scorer.weights_proj.weight.copy_(reference.scorer.weights_proj.weight)

  torch.testing.assert_close(
    layer.compress(hidden_states),
    _reference_indexer_compress(reference, hidden_states, position_ids),
  )
  torch.testing.assert_close(
    layer(hidden_states, q_residual, position_ids.squeeze(0)),
    reference(
      hidden_states,
      q_residual,
      position_ids,
      past_key_values=None,
      layer_idx=0,
    ),
  )


def test_deepseek_v4_indexer_backpropagates(random_input) -> None:
  layer = DeepSeekV4Indexer(
    d_model=8,
    q_lora_rank=5,
    compress_rate=2,
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
  )
  hidden_states = random_input(2, 6, 8).requires_grad_()
  compressed = layer.compress(hidden_states)

  compressed.square().mean().backward()

  assert hidden_states.grad is not None
  assert layer.kv_proj.weight.grad is not None
  assert layer.gate_proj.weight.grad is not None


def test_deepseek_v4_hca_compressor_matches_transformers(random_input) -> None:
  config = _deepseek_v4_config()
  reference = DeepseekV4HCACompressor(config)
  layer = DeepSeekV4HCACompressor(
    d_model=8,
    head_dim=4,
    compress_rate=2,
    rope_head_dim=4,
    rope_base=10_000.0,
    norm_eps=1e-6,
  )
  hidden_states = random_input(2, 6, 8)
  position_ids = torch.arange(6).unsqueeze(0).expand(2, -1)

  with torch.no_grad():
    reference.position_bias.copy_(
      torch.linspace(-0.2, 0.2, steps=reference.position_bias.numel()).view_as(
        reference.position_bias,
      )
    )
    layer.kv_proj.weight.copy_(reference.kv_proj.weight)
    layer.gate_proj.weight.copy_(reference.gate_proj.weight)
    layer.position_bias.copy_(reference.position_bias)
    layer.kv_norm.weight.copy_(reference.kv_norm.weight)

  actual_kv, actual_bias = layer(hidden_states, position_ids.squeeze(0))
  expected_kv, expected_bias = reference(
    hidden_states,
    q_residual=random_input(2, 6, 5),
    position_ids=position_ids,
    past_key_values=None,
    layer_idx=0,
  )

  torch.testing.assert_close(layer.compress(hidden_states), expected_kv.squeeze(1))
  torch.testing.assert_close(actual_kv, expected_kv)
  torch.testing.assert_close(actual_bias, expected_bias)


def test_deepseek_v4_hca_compressor_backpropagates(random_input) -> None:
  layer = DeepSeekV4HCACompressor(
    d_model=8,
    head_dim=4,
    compress_rate=2,
    rope_head_dim=4,
  )
  hidden_states = random_input(2, 6, 8).requires_grad_()
  compressed = layer.compress(hidden_states)

  compressed.square().mean().backward()

  assert hidden_states.grad is not None
  assert layer.kv_proj.weight.grad is not None
  assert layer.gate_proj.weight.grad is not None


def test_deepseek_v4_csa_compressor_matches_transformers(random_input) -> None:
  config = _deepseek_v4_config()
  reference = DeepseekV4CSACompressor(config)
  layer = DeepSeekV4CSACompressor(
    d_model=8,
    head_dim=4,
    q_lora_rank=5,
    compress_rate=2,
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    rope_head_dim=4,
    rope_base=10_000.0,
    norm_eps=1e-6,
  )
  hidden_states = random_input(2, 6, 8)
  q_residual = random_input(2, 6, 5)
  position_ids = torch.arange(6).unsqueeze(0).expand(2, -1)

  _copy_csa_compressor_weights(reference, layer)

  actual_kv, actual_bias = layer(hidden_states, q_residual, position_ids.squeeze(0))
  expected_kv, expected_bias = reference(
    hidden_states,
    q_residual=q_residual,
    position_ids=position_ids,
    past_key_values=None,
    layer_idx=0,
  )

  torch.testing.assert_close(layer.compress(hidden_states), expected_kv.squeeze(1))
  torch.testing.assert_close(actual_kv, expected_kv)
  torch.testing.assert_close(actual_bias, expected_bias)


def test_deepseek_v4_csa_compressor_backpropagates(random_input) -> None:
  layer = DeepSeekV4CSACompressor(
    d_model=8,
    head_dim=4,
    q_lora_rank=5,
    compress_rate=2,
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    rope_head_dim=4,
  )
  hidden_states = random_input(2, 6, 8).requires_grad_()
  compressed = layer.compress(hidden_states)

  compressed.square().mean().backward()

  assert hidden_states.grad is not None
  assert layer.kv_proj.weight.grad is not None
  assert layer.gate_proj.weight.grad is not None


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
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    compress_rates={
      "compressed_sparse_attention": 2,
      "heavily_compressed_attention": 2,
    },
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


def _reference_indexer_compress(
  reference: DeepseekV4Indexer,
  hidden_states: torch.Tensor,
  position_ids: torch.Tensor,
) -> torch.Tensor:
  batch, _, _ = hidden_states.shape
  kv = reference.kv_proj(hidden_states)
  gate = reference.gate_proj(hidden_states)
  usable = (kv.shape[1] // reference.compress_rate) * reference.compress_rate
  chunk_kv = kv[:, :usable]
  chunk_gate = gate[:, :usable]
  n_windows = chunk_kv.shape[1] // reference.compress_rate
  ratio = reference.compress_rate
  chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
  chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + reference.position_bias

  new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, reference.head_dim))
  new_gate = chunk_gate.new_full(
    (batch, n_windows, 2 * ratio, reference.head_dim),
    float("-inf"),
  )
  new_kv[:, :, ratio:] = chunk_kv[..., reference.head_dim :]
  new_gate[:, :, ratio:] = chunk_gate[..., reference.head_dim :]
  if n_windows > 1:
    new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : reference.head_dim]
    new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : reference.head_dim]

  compressed = reference.kv_norm(
    (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(
      dim=2,
    )
  )
  positions = torch.arange(n_windows, device=compressed.device)
  positions = positions * reference.compress_rate
  positions = positions.unsqueeze(0).expand(batch, -1)
  cos, sin = reference.rotary_emb(
    compressed,
    position_ids=positions,
    layer_type="compress",
  )
  return apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)


def _copy_csa_compressor_weights(
  reference: DeepseekV4CSACompressor,
  layer: DeepSeekV4CSACompressor,
) -> None:
  with torch.no_grad():
    reference.position_bias.copy_(
      torch.linspace(-0.2, 0.2, steps=reference.position_bias.numel()).view_as(
        reference.position_bias,
      )
    )
    reference.indexer.position_bias.copy_(
      torch.linspace(
        -0.1,
        0.1,
        steps=reference.indexer.position_bias.numel(),
      ).view_as(reference.indexer.position_bias)
    )
    layer.kv_proj.weight.copy_(reference.kv_proj.weight)
    layer.gate_proj.weight.copy_(reference.gate_proj.weight)
    layer.position_bias.copy_(reference.position_bias)
    layer.kv_norm.weight.copy_(reference.kv_norm.weight)
    layer.indexer.kv_proj.weight.copy_(reference.indexer.kv_proj.weight)
    layer.indexer.gate_proj.weight.copy_(reference.indexer.gate_proj.weight)
    layer.indexer.position_bias.copy_(reference.indexer.position_bias)
    layer.indexer.kv_norm.weight.copy_(reference.indexer.kv_norm.weight)
    layer.indexer.q_b_proj.weight.copy_(reference.indexer.q_b_proj.weight)
    layer.indexer.scorer.weights_proj.weight.copy_(
      reference.indexer.scorer.weights_proj.weight,
    )
