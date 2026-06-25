import pytest
import torch
from flm_modules import CausalSelfAttention


def test_causal_self_attention_preserves_input_shape(
  random_input,
) -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2, dropout=0.0)
  x = random_input(3, 5, 8)

  y = layer(x)

  assert y.shape == x.shape


def test_causal_self_attention_matches_saved_output(random_input) -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2, dropout=0.0)
  x = random_input(3, 5, 8)

  y = layer(x)

  torch.testing.assert_close(
    y[0, 0],
    torch.tensor(
      [
        -0.15856203436851501,
        -0.2611512839794159,
        -0.48171690106391907,
        -0.2562934458255768,
        -0.48398783802986145,
        -0.5388339161872864,
        0.24903729557991028,
        0.2406696379184723,
      ]
    ),
  )


def test_causal_self_attention_supports_bias_variant() -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2, bias=True)

  assert layer.qkv.bias is not None
  assert layer.out.bias is not None


def test_causal_self_attention_disables_dropout_in_eval(random_input) -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2, dropout=0.8)
  x = random_input(2, 4, 8)
  layer.eval()

  y1 = layer(x)
  y2 = layer(x)

  torch.testing.assert_close(y1, y2)


def test_causal_self_attention_rejects_invalid_head_count() -> None:
  with pytest.raises(ValueError, match="d_model must be divisible"):
    CausalSelfAttention(d_model=10, n_heads=3)
