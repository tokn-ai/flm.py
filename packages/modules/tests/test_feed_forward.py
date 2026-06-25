import torch
from flm_modules import SwiGLU
from torch.nn import functional as F


def test_swiglu_preserves_model_dimension(random_input) -> None:
  layer = SwiGLU(d_model=6, hidden_dim=10, dropout=0.0)
  x = random_input(2, 4, 6)

  y = layer(x)

  assert y.shape == x.shape


def test_swiglu_matches_manual_computation_without_dropout(random_input) -> None:
  layer = SwiGLU(d_model=4, hidden_dim=5, dropout=0.0, bias=True)
  x = random_input(2, 3, 4)

  gate, value = layer.up(x).chunk(2, dim=-1)
  expected = layer.down(F.silu(gate) * value)

  torch.testing.assert_close(layer(x), expected)


def test_swiglu_matches_saved_output(random_input) -> None:
  layer = SwiGLU(d_model=4, hidden_dim=5, dropout=0.0, bias=True)
  x = random_input(2, 3, 4)

  y = layer(x)

  torch.testing.assert_close(
    y[0, 0],
    torch.tensor(
      [
        0.31899863481521606,
        0.13644427061080933,
        0.4924498498439789,
        -0.43727385997772217,
      ]
    ),
  )


def test_swiglu_dropout_variant_is_deterministic_in_eval(random_input) -> None:
  layer = SwiGLU(d_model=4, hidden_dim=5, dropout=0.9)
  x = random_input(2, 3, 4)
  layer.eval()

  torch.testing.assert_close(layer(x), layer(x))
