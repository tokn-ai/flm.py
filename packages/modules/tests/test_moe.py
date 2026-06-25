import pytest
import torch
from flm_modules import DeepSeekMoE, DeepSeekTopKRouter, DeepSeekV4TopKRouter, SwiGLU
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter


def test_deepseek_moe_preserves_model_dimension(random_input) -> None:
  layer = DeepSeekMoE(
    d_model=6,
    d_ff=8,
    n_routed_experts=4,
    n_shared_experts=1,
    n_experts_per_token=2,
    n_group=2,
    topk_group=1,
  )
  x = random_input(2, 3, 6)

  y = layer(x)

  assert y.shape == x.shape


def test_deepseek_router_returns_flat_float32_logits(random_input) -> None:
  router = DeepSeekTopKRouter(d_model=4, n_routed_experts=3)
  x = random_input(2, 5, 4).half()

  logits = router(x)

  assert logits.shape == (10, 3)
  assert logits.dtype == torch.float32


def test_deepseek_moe_shared_experts_add_to_routed_output(random_input) -> None:
  layer = DeepSeekMoE(
    d_model=4,
    d_ff=5,
    n_routed_experts=2,
    n_shared_experts=1,
    n_experts_per_token=1,
  )
  shared = SwiGLU(d_model=4, d_ff=5)
  x = random_input(2, 3, 4)

  with torch.no_grad():
    for expert in layer.experts:
      for param in expert.parameters():
        param.zero_()
    shared.load_state_dict(layer.shared_experts.state_dict())

  torch.testing.assert_close(layer(x), shared(x))


def test_deepseek_moe_matches_transformers_route_tokens() -> None:
  config = DeepseekV3Config(
    hidden_size=4,
    intermediate_size=5,
    moe_intermediate_size=5,
    n_routed_experts=8,
    num_local_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    n_group=4,
    topk_group=2,
    norm_topk_prob=True,
    routed_scaling_factor=1.25,
  )
  reference = DeepseekV3MoE(config)
  layer = DeepSeekMoE(
    d_model=4,
    d_ff=5,
    n_routed_experts=8,
    n_shared_experts=1,
    n_experts_per_token=2,
    n_group=4,
    topk_group=2,
    norm_topk_prob=True,
    routed_scaling_factor=1.25,
  )
  router_logits = torch.tensor(
    [
      [0.1, -0.2, 0.5, 0.4, 0.3, -0.1, 0.9, 0.2],
      [-0.3, 0.8, 0.7, -0.4, 0.6, 0.5, -0.2, 0.1],
    ]
  )

  with torch.no_grad():
    reference.gate.e_score_correction_bias.copy_(layer.gate.e_score_correction_bias)

  actual_indices, actual_weights = layer.route_tokens_to_experts(router_logits)
  expected_indices, expected_weights = reference.route_tokens_to_experts(router_logits)

  torch.testing.assert_close(actual_indices, expected_indices)
  torch.testing.assert_close(actual_weights, expected_weights)


def test_deepseek_v4_topk_router_matches_transformers(random_input) -> None:
  config = DeepseekV4Config(
    hidden_size=4,
    n_routed_experts=4,
    num_experts_per_tok=2,
    routed_scaling_factor=1.25,
  )
  reference = DeepseekV4TopKRouter(config)
  layer = DeepSeekV4TopKRouter(
    d_model=4,
    n_routed_experts=4,
    n_experts_per_token=2,
    routed_scaling_factor=1.25,
  )
  x = random_input(2, 3, 4)

  with torch.no_grad():
    layer.weight.copy_(reference.weight)
    layer.e_score_correction_bias.copy_(reference.e_score_correction_bias)

  actual_logits, actual_weights, actual_indices = layer(x)
  expected_logits, expected_weights, expected_indices = reference(x)

  torch.testing.assert_close(actual_logits, expected_logits)
  torch.testing.assert_close(actual_indices, expected_indices)
  torch.testing.assert_close(actual_weights, expected_weights)


def test_deepseek_moe_matches_manual_expert_routing(random_input) -> None:
  layer = DeepSeekMoE(
    d_model=4,
    d_ff=5,
    n_routed_experts=3,
    n_shared_experts=0,
    n_experts_per_token=2,
    norm_topk_prob=False,
  )
  x = random_input(2, 3, 4)
  flat_x = x.reshape(-1, 4)
  router_logits = layer.gate(x)
  topk_indices, topk_weights = layer.route_tokens_to_experts(router_logits)

  expected = torch.zeros_like(flat_x)
  for token_idx, token in enumerate(flat_x):
    for topk_pos in range(topk_indices.shape[1]):
      expert_idx = topk_indices[token_idx, topk_pos]
      expected[token_idx] += (
        layer.experts[expert_idx](token.unsqueeze(0)).squeeze(0)
        * topk_weights[token_idx, topk_pos]
      )

  torch.testing.assert_close(layer(x), expected.view_as(x))


def test_deepseek_moe_backpropagates_to_used_experts_and_router(random_input) -> None:
  layer = DeepSeekMoE(
    d_model=4,
    d_ff=5,
    n_routed_experts=3,
    n_shared_experts=1,
    n_experts_per_token=2,
  )
  x = random_input(2, 3, 4).requires_grad_()

  layer(x).square().mean().backward()

  assert x.grad is not None
  assert layer.gate.weight.grad is not None
  assert layer.shared_experts.up.weight.grad is not None
  assert any(expert.up.weight.grad is not None for expert in layer.experts)


def test_deepseek_moe_rejects_invalid_grouping() -> None:
  with pytest.raises(ValueError, match="n_group must divide"):
    DeepSeekMoE(
      d_model=4,
      d_ff=5,
      n_routed_experts=3,
      n_shared_experts=1,
      n_experts_per_token=1,
      n_group=2,
    )


def test_deepseek_moe_rejects_single_expert_groups() -> None:
  with pytest.raises(ValueError, match="at least two experts"):
    DeepSeekMoE(
      d_model=4,
      d_ff=5,
      n_routed_experts=2,
      n_shared_experts=1,
      n_experts_per_token=1,
      n_group=2,
    )


def test_deepseek_moe_rejects_more_experts_than_selected_groups() -> None:
  with pytest.raises(ValueError, match="exceeds selected expert groups"):
    DeepSeekMoE(
      d_model=4,
      d_ff=5,
      n_routed_experts=4,
      n_shared_experts=1,
      n_experts_per_token=3,
      n_group=2,
      topk_group=1,
    )
