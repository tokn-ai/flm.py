import pytest
import torch
from flm_modules import GroupedLinear
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4GroupedLinear


def test_grouped_linear_preserves_grouped_shape(random_input) -> None:
  layer = GroupedLinear(
    in_features_per_group=3,
    out_features=8,
    n_groups=2,
  )
  x = random_input(4, 5, 2, 3)

  y = layer(x)

  assert y.shape == (4, 5, 2, 4)


def test_grouped_linear_matches_transformers_deepseek_v4(random_input) -> None:
  reference = DeepseekV4GroupedLinear(
    in_features_per_group=3,
    out_features=8,
    n_groups=2,
  )
  layer = GroupedLinear(
    in_features_per_group=3,
    out_features=8,
    n_groups=2,
  )
  x = random_input(4, 5, 2, 3)

  with torch.no_grad():
    layer.weight.copy_(reference.weight)

  torch.testing.assert_close(layer(x), reference(x))


def test_grouped_linear_rejects_invalid_group_count() -> None:
  with pytest.raises(ValueError, match="n_groups must be positive"):
    GroupedLinear(
      in_features_per_group=3,
      out_features=8,
      n_groups=0,
    )


def test_grouped_linear_rejects_wrong_input_groups(random_input) -> None:
  layer = GroupedLinear(
    in_features_per_group=3,
    out_features=8,
    n_groups=2,
  )
  x = random_input(4, 5, 3, 3)

  with pytest.raises(ValueError, match="group dimension"):
    layer(x)
