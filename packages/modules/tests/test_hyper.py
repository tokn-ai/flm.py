import torch
from flm_modules import (
  DeepSeekV4HyperConnection,
  DeepSeekV4HyperHead,
  UnweightedRMSNorm,
)
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
  DeepseekV4HyperConnection,
  DeepseekV4HyperHead,
  DeepseekV4UnweightedRMSNorm,
)


def test_unweighted_rms_norm_matches_transformers(random_input) -> None:
  layer = UnweightedRMSNorm(eps=1e-6)
  reference = DeepseekV4UnweightedRMSNorm(eps=1e-6)
  x = random_input(2, 3, 4)

  torch.testing.assert_close(layer(x), reference(x))


def test_deepseek_v4_hyper_connection_matches_transformers(random_input) -> None:
  config = DeepseekV4Config(
    hidden_size=4,
    hc_mult=3,
    hc_sinkhorn_iters=3,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
  )
  reference = DeepseekV4HyperConnection(config)
  layer = DeepSeekV4HyperConnection(
    d_model=4,
    hc_mult=3,
    hc_sinkhorn_iters=3,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
  )
  x = random_input(2, 5, 3, 4)

  with torch.no_grad():
    reference.fn.copy_(
      torch.linspace(-0.2, 0.2, steps=reference.fn.numel()).view_as(reference.fn)
    )
    reference.base.copy_(torch.linspace(-0.1, 0.1, steps=reference.base.numel()))
    reference.scale.copy_(torch.tensor([0.5, -0.25, 0.75]))
    layer.fn.copy_(reference.fn)
    layer.base.copy_(reference.base)
    layer.scale.copy_(reference.scale)

  actual_post, actual_comb, actual_collapsed = layer(x)
  expected_post, expected_comb, expected_collapsed = reference(x)

  torch.testing.assert_close(actual_post, expected_post)
  torch.testing.assert_close(actual_comb, expected_comb)
  torch.testing.assert_close(actual_collapsed, expected_collapsed)


def test_deepseek_v4_hyper_head_matches_transformers(random_input) -> None:
  config = DeepseekV4Config(
    hidden_size=4,
    hc_mult=3,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
  )
  reference = DeepseekV4HyperHead(config)
  layer = DeepSeekV4HyperHead(
    d_model=4,
    hc_mult=3,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
  )
  x = random_input(2, 5, 3, 4)

  with torch.no_grad():
    reference.hc_fn.copy_(
      torch.linspace(-0.2, 0.2, steps=reference.hc_fn.numel()).view_as(reference.hc_fn)
    )
    reference.hc_base.copy_(torch.linspace(-0.1, 0.1, steps=reference.hc_base.numel()))
    reference.hc_scale.copy_(torch.tensor([0.5]))
    layer.hc_fn.copy_(reference.hc_fn)
    layer.hc_base.copy_(reference.hc_base)
    layer.hc_scale.copy_(reference.hc_scale)

  torch.testing.assert_close(layer(x), reference(x))
