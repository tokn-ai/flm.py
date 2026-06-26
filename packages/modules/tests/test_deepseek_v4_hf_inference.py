import importlib.util
import sys
import types
from pathlib import Path

import torch
from flm_modules import DeepSeekTopKRouter, DeepSeekV4MLP, RMSNorm, RouterScoring

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HF_INFERENCE_PATH = _REPO_ROOT / "huggingface_models" / "deepseek_v4_inference.py"


def test_rms_norm_matches_hf_inference_model_bfloat16(random_input) -> None:
  hf_model = _load_hf_inference_model()
  reference = hf_model.RMSNorm(4, eps=1e-6)
  layer = RMSNorm(d_model=4, eps=1e-6)
  x = random_input(2, 3, 4).bfloat16()

  with torch.no_grad():
    reference.weight.copy_(torch.tensor([1.0, 0.5, 2.0, -1.0]))
    layer.weight.copy_(reference.weight)

  actual = layer(x)
  expected = reference(x)

  assert actual.dtype == torch.bfloat16
  torch.testing.assert_close(actual, expected)


def test_deepseek_v4_mlp_matches_hf_inference_expert(random_input) -> None:
  hf_model = _load_hf_inference_model()
  reference = hf_model.Expert(4, 5, dtype=torch.float32, swiglu_limit=2.0)
  layer = DeepSeekV4MLP(d_model=4, d_ff=5, swiglu_limit=2.0)
  x = random_input(2, 3, 4)

  with torch.no_grad():
    reference.w1.weight.copy_(torch.linspace(-0.2, 0.2, steps=20).view(5, 4))
    reference.w3.weight.copy_(torch.linspace(0.2, -0.2, steps=20).view(5, 4))
    reference.w2.weight.copy_(torch.linspace(-0.1, 0.1, steps=20).view(4, 5))
    layer.gate_proj.weight.copy_(reference.w1.weight)
    layer.up_proj.weight.copy_(reference.w3.weight)
    layer.down_proj.weight.copy_(reference.w2.weight)

  torch.testing.assert_close(layer(x), reference(x))


def test_deepseek_v4_router_matches_hf_inference_gate(random_input) -> None:
  hf_model = _load_hf_inference_model()
  args = hf_model.ModelArgs(
    vocab_size=32,
    dim=4,
    n_hash_layers=0,
    n_routed_experts=4,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    route_scale=1.25,
  )
  reference = hf_model.Gate(layer_id=0, args=args)
  layer = DeepSeekTopKRouter(
    d_model=4,
    n_routed_experts=4,
    n_experts_per_token=2,
    scoring_func=RouterScoring.SQRT_SOFTPLUS,
    routed_scaling_factor=1.25,
    grouped_topk=False,
  )
  x = random_input(6, 4)

  with torch.no_grad():
    reference.weight.copy_(torch.linspace(-0.5, 0.5, steps=16).view(4, 4))
    reference.bias.copy_(torch.tensor([0.1, -0.2, 0.0, 0.3]))
    layer.weight.copy_(reference.weight)
    layer.e_score_correction_bias.copy_(reference.bias)

  expected_weights, expected_indices = reference(x)
  actual_indices, actual_weights = layer.route_logits(layer(x))

  torch.testing.assert_close(actual_indices, expected_indices)
  torch.testing.assert_close(actual_weights, expected_weights)


def _load_hf_inference_model():
  _install_kernel_stub()
  module_name = "hf_deepseek_v4_flash_inference_model"
  spec = importlib.util.spec_from_file_location(module_name, _HF_INFERENCE_PATH)
  module = importlib.util.module_from_spec(spec)
  if spec is None or spec.loader is None:
    raise RuntimeError("failed to load Hugging Face DeepSeek-V4 inference model")
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def _install_kernel_stub() -> None:
  if "kernel" in sys.modules:
    return

  kernel = types.ModuleType("kernel")

  def unavailable(*_args, **_kwargs):
    raise RuntimeError("DeepSeek-V4 inference kernel stub should not be called")

  for name in [
    "act_quant",
    "fp4_act_quant",
    "fp8_gemm",
    "fp4_gemm",
    "sparse_attn",
    "hc_split_sinkhorn",
  ]:
    setattr(kernel, name, unavailable)
  sys.modules["kernel"] = kernel
