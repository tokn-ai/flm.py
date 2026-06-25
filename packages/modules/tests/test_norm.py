import torch
from flm_modules import RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def test_rms_norm_matches_manual_computation(random_input) -> None:
  layer = RMSNorm(d_model=4, eps=1e-6)
  x = random_input(2, 3, 4)

  expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + layer.eps)

  torch.testing.assert_close(layer(x), expected)


def test_rms_norm_matches_saved_output(random_input) -> None:
  layer = RMSNorm(d_model=4, eps=1e-6)
  x = random_input(2, 3, 4)

  y = layer(x)

  torch.testing.assert_close(
    y[0, 0],
    torch.tensor(
      [
        1.1531217098236084,
        0.8900336623191833,
        0.5390151739120483,
        -1.2600045204162598,
      ]
    ),
  )


def test_rms_norm_applies_learned_weight() -> None:
  layer = RMSNorm(d_model=3, eps=0.0)
  layer.weight.data = torch.tensor([1.0, 2.0, 3.0])
  x = torch.ones(2, 3)

  y = layer(x)

  torch.testing.assert_close(y, layer.weight.expand_as(x))


def test_rms_norm_matches_transformers_llama_rms_norm(random_input) -> None:
  layer = RMSNorm(d_model=4, eps=1e-6)
  reference = LlamaRMSNorm(hidden_size=4, eps=1e-6)
  x = random_input(2, 3, 4)

  with torch.no_grad():
    reference.weight.copy_(layer.weight)

  torch.testing.assert_close(layer(x), reference(x))
