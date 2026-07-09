import torch
from flm_modules import Muon, configure_adamw, configure_muon
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


def test_configure_muon_groups_matrix_and_vector_parameters() -> None:
  model = TinyModel()

  optimizer = configure_muon(
    model,
    learning_rate=1e-3,
    weight_decay=0.2,
    momentum=0.8,
    nesterov=False,
    ns_steps=3,
    adamw_betas=(0.7, 0.9),
    adamw_eps=1e-7,
  )

  assert isinstance(optimizer, Muon)
  muon_params = set(optimizer.param_groups[0]["params"])
  adamw_params = set(optimizer.param_groups[1]["params"])
  assert optimizer.param_groups[0]["use_muon"] is True
  assert optimizer.param_groups[1]["use_muon"] is False
  assert optimizer.param_groups[0]["lr"] == 1e-3
  assert optimizer.param_groups[0]["momentum"] == 0.8
  assert optimizer.param_groups[0]["nesterov"] is False
  assert optimizer.param_groups[0]["ns_steps"] == 3
  assert optimizer.param_groups[1]["adamw_betas"] == (0.7, 0.9)
  assert optimizer.param_groups[1]["adamw_eps"] == 1e-7
  assert optimizer.param_groups[0]["weight_decay"] == 0.2
  assert optimizer.param_groups[1]["weight_decay"] == 0.0
  assert model.linear.weight in muon_params
  assert model.linear.bias in adamw_params
  assert model.norm.weight in adamw_params
  assert model.norm.bias in adamw_params
  assert model.frozen not in muon_params
  assert model.frozen not in adamw_params


def test_muon_step_updates_muon_and_adamw_parameters() -> None:
  torch.manual_seed(0)
  model = TinyModel()
  optimizer = configure_muon(model, learning_rate=1e-2, weight_decay=0.0)
  before_weight = model.linear.weight.detach().clone()
  before_bias = model.linear.bias.detach().clone()

  output = model.linear(torch.ones(4, 3)).square().sum()
  output.backward()
  optimizer.step()

  assert not torch.equal(model.linear.weight, before_weight)
  assert not torch.equal(model.linear.bias, before_bias)
  assert "momentum_buffer" in optimizer.state[model.linear.weight]
  assert "exp_avg" in optimizer.state[model.linear.bias]
  assert "exp_avg_sq" in optimizer.state[model.linear.bias]
