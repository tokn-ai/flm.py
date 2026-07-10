import pytest
import torch
from flm_modules.rope import (
  RopeLayout,
  RotaryEmbedding,
  SpeedrunYaRN,
  apply_rotary,
  rotate_half,
)
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
  LlamaRotaryEmbedding,
  apply_rotary_pos_emb,
)


def test_rotate_half_rotates_hidden_dimension_halves() -> None:
  x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

  y = rotate_half(x)

  torch.testing.assert_close(y, torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))


def test_rotate_half_supports_interleaved_layout() -> None:
  x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

  y = rotate_half(x, layout=RopeLayout.INTERLEAVED)

  torch.testing.assert_close(y, torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]]))


def test_apply_rotary_identity_when_sin_is_zero(random_input) -> None:
  x = random_input(2, 3, 4, 6)
  cos = torch.ones(4, 6)
  sin = torch.zeros(4, 6)

  y = apply_rotary(x, cos, sin)

  torch.testing.assert_close(y, x)


def test_apply_rotary_identity_supports_interleaved_layout(random_input) -> None:
  x = random_input(2, 3, 4, 6)
  cos = torch.ones(4, 6)
  sin = torch.zeros(4, 6)

  y = apply_rotary(x, cos, sin, layout=RopeLayout.INTERLEAVED)

  torch.testing.assert_close(y, x)


def test_apply_rotary_supports_deepseek_v32_layout() -> None:
  x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
  cos = torch.tensor([[0.5, 0.25, 0.5, 0.25]])
  sin = torch.tensor([[0.1, 0.2, 0.1, 0.2]])

  y = apply_rotary(x, cos, sin, layout=RopeLayout.DEEPSEEK_V32)

  torch.testing.assert_close(
    y,
    torch.tensor([[[[0.3, -0.05, 1.1, 1.6]]]]),
  )


def test_rotary_embedding_returns_cos_sin_shapes(random_input) -> None:
  rope = RotaryEmbedding(dim=6)
  q = random_input(2, 3, 4, 6)

  cos, sin = rope(q)

  assert cos.shape == (4, 6)
  assert sin.shape == (4, 6)


def test_rotary_embedding_supports_explicit_positions_variant(random_input) -> None:
  rope = RotaryEmbedding(dim=4)
  q = random_input(1, 2, 3, 4)

  default_cos, default_sin = rope(q)
  explicit_cos, explicit_sin = rope(q, positions=torch.arange(3))

  torch.testing.assert_close(default_cos, explicit_cos)
  torch.testing.assert_close(default_sin, explicit_sin)


def test_rotary_embedding_matches_saved_default_output(random_input) -> None:
  rope = RotaryEmbedding(dim=4)
  q = random_input(1, 2, 3, 4)

  cos, sin = rope(q)
  q_out = apply_rotary(q, cos, sin, layout=rope.layout)

  torch.testing.assert_close(
    q_out[0, 0, 1],
    torch.tensor(
      [
        0.4027911126613617,
        -1.2184367179870605,
        0.5475999712944031,
        -1.6169319152832031,
      ]
    ),
  )


def test_rotary_embedding_matches_saved_explicit_positions_output(
  random_input,
) -> None:
  rope = RotaryEmbedding(dim=4)
  q = random_input(1, 2, 3, 4)

  cos, sin = rope(q, positions=torch.tensor([2, 3, 4]))
  q_out = apply_rotary(q, cos, sin, layout=rope.layout)

  torch.testing.assert_close(
    q_out[0, 0, 1],
    torch.tensor(
      [
        -0.665551483631134,
        -1.185856580734253,
        0.13837489485740662,
        -1.6409757137298584,
      ]
    ),
  )


def test_rotary_embedding_rejects_odd_dimensions() -> None:
  with pytest.raises(ValueError, match="must be even"):
    RotaryEmbedding(dim=5)


def test_rotary_embedding_matches_transformers_llama_rope(random_input) -> None:
  config = LlamaConfig(
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=16,
    rope_theta=10_000.0,
  )
  reference = LlamaRotaryEmbedding(config)
  rope = RotaryEmbedding(dim=4, base=10_000.0)
  q = random_input(1, 2, 3, 4)
  k = random_input(1, 2, 3, 4)
  position_ids = torch.arange(3).unsqueeze(0)

  cos, sin = reference(q, position_ids)
  expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
  cos, sin = rope(q)
  q_out = apply_rotary(q, cos, sin, layout=rope.layout)
  k_out = apply_rotary(k, cos, sin, layout=rope.layout)

  torch.testing.assert_close(q_out, expected_q)
  torch.testing.assert_close(k_out, expected_k)


def test_rotary_embedding_supports_interleaved_layout(random_input) -> None:
  rope = RotaryEmbedding(dim=4, layout=RopeLayout.INTERLEAVED)
  q = random_input(1, 2, 3, 4)

  cos, sin = rope(q)
  q_out = apply_rotary(q, cos, sin, layout=rope.layout)

  assert q_out.shape == q.shape
  torch.testing.assert_close(
    q_out[0, 0, 1],
    torch.tensor(
      [
        1.4053846597671509,
        -0.09615802764892578,
        -0.027018923312425613,
        -1.6050174236297607,
      ]
    ),
  )


def test_rotary_embedding_supports_batched_positions(random_input) -> None:
  rope = RotaryEmbedding(dim=4)
  x = random_input(2, 3, 4)
  positions = torch.tensor([[0, 1, 2], [2, 3, 4]])

  cos, sin = rope(x, positions=positions)

  assert cos.shape == (2, 3, 4)
  assert sin.shape == (2, 3, 4)
  torch.testing.assert_close(cos[0, 1], rope(x[:1], positions=torch.tensor([1]))[0][0])
  torch.testing.assert_close(cos[1, 0], rope(x[:1], positions=torch.tensor([2]))[0][0])


def test_apply_rotary_supports_partial_trailing_dimension(random_input) -> None:
  x = random_input(1, 2, 3, 6)
  cos = torch.ones(3, 4)
  sin = torch.zeros(3, 4)

  y = apply_rotary(x, cos, sin, rotary_dim=4)

  torch.testing.assert_close(y, x)


def test_speedrun_yarn_half_truncates_rotary_dimensions(random_input) -> None:
  yarn = SpeedrunYaRN(head_dim=8, max_seq_len=16)
  x = random_input(2, 3, 5, 8)

  output = yarn(x)

  assert output.shape == x.shape
  torch.testing.assert_close(output[..., 4:], x[..., 4:])


def test_speedrun_yarn_updates_frequency_and_attention_scale() -> None:
  yarn = SpeedrunYaRN(head_dim=8, max_seq_len=16)
  before_frequency = yarn.angular_freq.clone()
  before_scale = yarn.attention_scale

  yarn.apply_window_change(4, 8)

  assert not torch.equal(yarn.angular_freq, before_frequency)
  assert yarn.attention_scale > before_scale


def test_speedrun_yarn_supports_paired_head_width(random_input) -> None:
  yarn = SpeedrunYaRN(head_dim=8, max_seq_len=16, paired=True)
  x = random_input(2, 3, 5, 16)

  assert yarn(x).shape == x.shape
