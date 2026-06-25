import torch
from flm_modules import configure_adamw
from torch import nn


class TinyModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.linear = nn.Linear(3, 2)
    self.norm = nn.LayerNorm(2)
    self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)


def test_configure_adamw_sets_hyperparameters() -> None:
  model = TinyModel()

  optimizer = configure_adamw(
    model,
    learning_rate=1e-3,
    weight_decay=0.2,
    betas=(0.8, 0.9),
    eps=1e-7,
  )

  assert isinstance(optimizer, torch.optim.AdamW)
  assert optimizer.param_groups[0]["lr"] == 1e-3
  assert optimizer.param_groups[0]["betas"] == (0.8, 0.9)
  assert optimizer.param_groups[0]["eps"] == 1e-7


def test_configure_adamw_groups_decay_and_no_decay_parameters() -> None:
  model = TinyModel()

  optimizer = configure_adamw(model, weight_decay=0.2)

  decay_params = set(optimizer.param_groups[0]["params"])
  no_decay_params = set(optimizer.param_groups[1]["params"])
  assert optimizer.param_groups[0]["weight_decay"] == 0.2
  assert optimizer.param_groups[1]["weight_decay"] == 0.0
  assert model.linear.weight in decay_params
  assert model.linear.bias in no_decay_params
  assert model.norm.weight in no_decay_params
  assert model.norm.bias in no_decay_params
  assert model.frozen not in decay_params
  assert model.frozen not in no_decay_params
