import pytest
import torch
from flm_modules.rope import RotaryEmbedding, apply_rotary, rotate_half


def test_rotate_half_rotates_even_odd_pairs() -> None:
  x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

  y = rotate_half(x)

  torch.testing.assert_close(y, torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]]))


def test_apply_rotary_identity_when_sin_is_zero(random_input) -> None:
  x = random_input(2, 3, 4, 6)
  cos = torch.ones(4, 6)
  sin = torch.zeros(4, 6)

  y = apply_rotary(x, cos, sin)

  torch.testing.assert_close(y, x)


def test_rotary_embedding_preserves_query_and_key_shapes(random_input) -> None:
  rope = RotaryEmbedding(dim=6)
  q = random_input(2, 3, 4, 6)
  k = random_input(2, 3, 4, 6)

  q_out, k_out = rope(q, k)

  assert q_out.shape == q.shape
  assert k_out.shape == k.shape


def test_rotary_embedding_supports_explicit_positions_variant(random_input) -> None:
  rope = RotaryEmbedding(dim=4)
  q = random_input(1, 2, 3, 4)
  k = random_input(1, 2, 3, 4)

  default_q, default_k = rope(q, k)
  explicit_q, explicit_k = rope(q, k, positions=torch.arange(3))

  torch.testing.assert_close(default_q, explicit_q)
  torch.testing.assert_close(default_k, explicit_k)


def test_rotary_embedding_matches_saved_default_output(random_input) -> None:
  rope = RotaryEmbedding(dim=4)
  q = random_input(1, 2, 3, 4)
  k = random_input(1, 2, 3, 4)

  q_out, _ = rope(q, k)

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


def test_rotary_embedding_matches_saved_explicit_positions_output(
  random_input,
) -> None:
  rope = RotaryEmbedding(dim=4)
  q = random_input(1, 2, 3, 4)
  k = random_input(1, 2, 3, 4)

  q_out, _ = rope(q, k, positions=torch.tensor([2, 3, 4]))

  torch.testing.assert_close(
    q_out[0, 0, 1],
    torch.tensor(
      [
        -0.49741023778915405,
        1.3179285526275635,
        0.005084685981273651,
        -1.6052367687225342,
      ]
    ),
  )


def test_rotary_embedding_rejects_odd_dimensions() -> None:
  with pytest.raises(ValueError, match="must be even"):
    RotaryEmbedding(dim=5)
