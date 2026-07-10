import pytest
import torch
from flm_modules import (
  CompositeOptimizer,
  Muon,
  NorMuon,
  configure_adamw,
  configure_muon,
  configure_normuon,
)
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


def test_configure_muon_builds_composite_for_matrix_and_vector_parameters() -> None:
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

  assert isinstance(optimizer, CompositeOptimizer)
  assert len(optimizer.optimizers) == 2
  muon_optimizer, adamw_optimizer = optimizer.optimizers
  assert isinstance(muon_optimizer, Muon)
  assert isinstance(adamw_optimizer, torch.optim.AdamW)
  muon_params = set(muon_optimizer.param_groups[0]["params"])
  adamw_params = set(adamw_optimizer.param_groups[0]["params"])
  assert optimizer.param_groups[0]["lr"] == 1e-3
  assert optimizer.param_groups[0]["momentum"] == 0.8
  assert optimizer.param_groups[0]["nesterov"] is False
  assert optimizer.param_groups[0]["ns_steps"] == 3
  assert optimizer.param_groups[1]["betas"] == (0.7, 0.9)
  assert optimizer.param_groups[1]["eps"] == 1e-7
  assert optimizer.param_groups[0]["weight_decay"] == 0.2
  assert optimizer.param_groups[1]["weight_decay"] == 0.0
  assert model.linear.weight in muon_params
  assert model.linear.bias in adamw_params
  assert model.norm.weight in adamw_params
  assert model.norm.bias in adamw_params
  assert model.frozen not in muon_params
  assert model.frozen not in adamw_params


def test_muon_rejects_non_matrix_parameters() -> None:
  with pytest.raises(ValueError, match="only supports 2D parameters"):
    Muon([nn.Parameter(torch.ones(2))])


def test_configure_normuon_builds_matrix_and_adam_fallback_optimizers() -> None:
  model = TinyModel()

  optimizer = configure_normuon(
    model,
    learning_rate=1e-3,
    weight_decay=0.2,
    momentum=0.85,
    beta2=0.9,
  )

  assert isinstance(optimizer, CompositeOptimizer)
  normuon, adamw = optimizer.optimizers
  assert isinstance(normuon, NorMuon)
  assert isinstance(adamw, torch.optim.AdamW)
  assert model.linear.weight in set(normuon.param_groups[0]["params"])
  assert model.linear.bias in set(adamw.param_groups[0]["params"])
  assert normuon.param_groups[0]["momentum"] == 0.85
  assert normuon.param_groups[0]["beta2"] == 0.9


def test_normuon_step_tracks_low_rank_variance_and_updates_parameter() -> None:
  parameter = nn.Parameter(torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]]))
  optimizer = NorMuon(
    [parameter],
    lr=0.03,
    momentum=0.9,
    beta2=0.95,
    weight_decay=0.01,
  )
  before = parameter.detach().clone()
  parameter.grad = torch.tensor([[0.2, -0.3, 0.4], [0.5, -0.6, 0.7]])

  optimizer.step()

  assert not torch.equal(parameter, before)
  state = optimizer.state[parameter]
  assert state["momentum_buffer"].shape == parameter.shape
  assert state["second_momentum_buffer"].shape == (1, 3)
  assert torch.isfinite(parameter).all()


def test_normuon_rejects_non_matrix_parameters() -> None:
  with pytest.raises(ValueError, match="only supports 2D"):
    NorMuon([nn.Parameter(torch.ones(2))])


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


def test_composite_optimizer_round_trips_state() -> None:
  torch.manual_seed(0)
  model = TinyModel()
  optimizer = configure_muon(model, learning_rate=1e-2, weight_decay=0.0)

  output = model.linear(torch.ones(4, 3)).square().sum()
  output.backward()
  optimizer.step()
  state = optimizer.state_dict()

  restored = TinyModel()
  restored_optimizer = configure_muon(restored, learning_rate=1e-2, weight_decay=0.0)
  restored_optimizer.load_state_dict(state)

  assert "momentum_buffer" in restored_optimizer.state[restored.linear.weight]
  assert "exp_avg" in restored_optimizer.state[restored.linear.bias]
  assert "exp_avg_sq" in restored_optimizer.state[restored.linear.bias]


def test_muon_matrix_step_matches_torch_muon() -> None:
  torch_muon = getattr(torch.optim, "Muon", None)
  if torch_muon is None:
    pytest.skip("torch.optim.Muon is not available")

  param = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
  grads = (
    torch.tensor([[0.2, -0.3, 0.4], [0.5, -0.6, 0.7]]),
    torch.tensor([[-0.1, 0.3, -0.5], [0.7, -0.9, 1.1]]),
  )
  ours = nn.Parameter(param.clone())
  expected = nn.Parameter(param.clone())
  kwargs = {
    "lr": 0.03,
    "weight_decay": 0.01,
    "momentum": 0.9,
    "nesterov": True,
    "ns_steps": 5,
  }
  ours_optimizer = Muon([ours], **kwargs)
  torch_optimizer = torch_muon([expected], **kwargs)

  for grad in grads:
    ours.grad = grad.clone()
    expected.grad = grad.clone()
    ours_optimizer.step()
    torch_optimizer.step()

  torch.testing.assert_close(ours, expected, rtol=0.0, atol=0.0)
