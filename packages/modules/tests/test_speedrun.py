import pytest
import torch
from flm_modules import (
  BigramHashEmbedding,
  MultiwayDynamicDenseConnections,
  TokenSmear,
)


def test_token_smear_is_identity_at_initialization(random_input) -> None:
  smear = TokenSmear(d_model=8, gate_dim=4)
  x = random_input(2, 5, 8)

  torch.testing.assert_close(smear(x), x)


def test_token_smear_injects_only_previous_tokens() -> None:
  smear = TokenSmear(d_model=4, gate_dim=2)
  with torch.no_grad():
    smear.scale.fill_(2.0)
  x = torch.arange(20, dtype=torch.float32).view(1, 5, 4)

  output = smear(x)

  torch.testing.assert_close(output[:, :1], x[:, :1])
  torch.testing.assert_close(output[:, 1:], x[:, 1:] + x[:, :-1])


def test_bigram_hash_embedding_matches_reference_hash() -> None:
  layer = BigramHashEmbedding(
    num_embeddings=101,
    embedding_dim=4,
    sign_table_rows=8,
  )
  input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
  modulus = 100
  expected = torch.tensor(
    [
      [
        modulus,
        (36_313 * 2 ^ 27_191 * 1) % modulus,
        (36_313 * 3 ^ 27_191 * 2) % modulus,
      ],
      [
        modulus,
        (36_313 * 5 ^ 27_191 * 4) % modulus,
        (36_313 * 6 ^ 27_191 * 5) % modulus,
      ],
    ]
  )

  torch.testing.assert_close(layer.hash_ids(input_ids), expected)


def test_bigram_hash_embedding_is_zero_at_initialization() -> None:
  layer = BigramHashEmbedding(num_embeddings=101, embedding_dim=4)
  input_ids = torch.tensor([[1, 2, 3]])

  torch.testing.assert_close(
    layer(input_ids),
    torch.zeros(1, 3, 4),
  )


def test_speedrun_modules_validate_shapes() -> None:
  smear = TokenSmear(d_model=8, gate_dim=4)
  bigram = BigramHashEmbedding(num_embeddings=101, embedding_dim=4)

  with pytest.raises(ValueError, match="batch, sequence"):
    smear(torch.ones(5, 8))
  with pytest.raises(ValueError, match="batch, sequence"):
    bigram(torch.ones(5, dtype=torch.long))


def test_mudd_starts_from_configured_biases(random_input) -> None:
  mudd = MultiwayDynamicDenseConnections(
    d_model=8,
    hidden_dim=4,
    max_coefficients=5,
    output_scale=0.1,
  )
  with torch.no_grad():
    mudd.bias[0, :3].copy_(torch.tensor([10.0, -5.0, 2.0]))
  x = random_input(2, 3, 8)

  coefficients = mudd(x, route=0, num_coefficients=3)

  assert len(coefficients) == 3
  torch.testing.assert_close(coefficients[0], torch.ones(2, 3, 1))
  torch.testing.assert_close(coefficients[1], -0.5 * torch.ones(2, 3, 1))
  torch.testing.assert_close(coefficients[2], 0.2 * torch.ones(2, 3, 1))


def test_mudd_validates_route_and_coefficient_count(random_input) -> None:
  mudd = MultiwayDynamicDenseConnections(d_model=8, hidden_dim=4)
  x = random_input(2, 3, 8)

  with pytest.raises(ValueError, match="route"):
    mudd(x, route=2, num_coefficients=1)
  with pytest.raises(ValueError, match="coefficients"):
    mudd(x, route=0, num_coefficients=15)
