import importlib.util
import sys
import types
from pathlib import Path

import torch
from flm_modules import DeepSeekDSAIndexer

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HF_INFERENCE_PATH = _REPO_ROOT / "huggingface_models" / "deepseek_v32_inference.py"


def test_deepseek_dsa_indexer_matches_hf_inference_indexer(random_input) -> None:
  hf_model = _load_hf_inference_model()
  args = hf_model.ModelArgs(
    max_batch_size=2,
    max_seq_len=8,
    dim=8,
    q_lora_rank=5,
    qk_rope_head_dim=4,
    index_n_heads=3,
    index_head_dim=128,
    index_topk=3,
    rope_theta=10_000.0,
    original_seq_len=8,
    rope_factor=1.0,
  )
  reference = hf_model.Indexer(args)
  layer = DeepSeekDSAIndexer(
    d_model=8,
    q_lora_rank=5,
    qk_rope_head_dim=4,
    index_n_heads=3,
    index_head_dim=128,
    index_topk=3,
    rope_base=10_000.0,
  ).to(torch.bfloat16)
  x = random_input(2, 5, 8).bfloat16()
  q_residual = random_input(2, 5, 5).bfloat16()
  positions = torch.arange(5).unsqueeze(0).expand(2, -1)
  freqs_cis = hf_model.precompute_freqs_cis(args)[:5]
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=torch.float32)
  generator = torch.Generator().manual_seed(7)

  with torch.no_grad():
    reference.wq_b.weight.copy_(
      torch.randn(3 * 128, 5, generator=generator, dtype=torch.bfloat16) * 0.1
    )
    reference.wk.weight.copy_(
      torch.randn(128, 8, generator=generator, dtype=torch.bfloat16) * 0.1
    )
    reference.k_norm.weight.copy_(
      torch.randn(128, generator=generator, dtype=torch.float32) * 0.1 + 1.0
    )
    reference.k_norm.bias.copy_(
      torch.randn(128, generator=generator, dtype=torch.float32) * 0.1
    )
    reference.weights_proj.weight.copy_(
      torch.randn(3, 8, generator=generator, dtype=torch.float32) * 0.1
    )
    layer.wq_b.weight.copy_(reference.wq_b.weight)
    layer.wk.weight.copy_(reference.wk.weight)
    layer.k_norm.weight.copy_(reference.k_norm.weight.to(torch.bfloat16))
    layer.k_norm.bias.copy_(reference.k_norm.bias.to(torch.bfloat16))
    layer.weights_proj.weight.copy_(reference.weights_proj.weight)

  try:
    expected = reference(
      x,
      q_residual,
      start_pos=0,
      freqs_cis=freqs_cis,
      mask=attention_mask,
    )

    torch.testing.assert_close(
      layer(x, q_residual, attention_mask=attention_mask, positions=positions),
      expected.to(torch.int32),
    )
  finally:
    hf_model.restore_test_stubs()


def _load_hf_inference_model():
  previous_kernel = sys.modules.get("kernel")
  previous_hadamard = sys.modules.get("fast_hadamard_transform")
  _install_kernel_stub()
  _install_hadamard_stub()
  module_name = "hf_deepseek_v32_exp_inference_model"
  spec = importlib.util.spec_from_file_location(module_name, _HF_INFERENCE_PATH)
  module = importlib.util.module_from_spec(spec)
  if spec is None or spec.loader is None:
    raise RuntimeError("failed to load Hugging Face DeepSeek-V3.2 inference model")
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  module.dist.broadcast = lambda tensor, src: tensor
  module.restore_test_stubs = lambda: (
    _restore_module("kernel", previous_kernel),
    _restore_module("fast_hadamard_transform", previous_hadamard),
  )
  return module


def _install_kernel_stub() -> None:
  kernel = types.ModuleType("kernel")

  def act_quant(x, _block_size, _scale_fmt=None):
    scale = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=torch.float32)
    return x, scale

  def fp8_gemm(*_args, **_kwargs):
    raise RuntimeError("DeepSeek-V3.2 fp8_gemm stub should not be called")

  def fp8_index(q, weights, k_cache, _k_scale_cache):
    scores = torch.matmul(q.float(), k_cache.float().transpose(-1, -2).unsqueeze(1))
    scores = scores.relu()
    return (scores * weights.float()).sum(dim=2)

  kernel.act_quant = act_quant
  kernel.fp8_gemm = fp8_gemm
  kernel.fp8_index = fp8_index
  sys.modules["kernel"] = kernel


def _install_hadamard_stub() -> None:
  module = types.ModuleType("fast_hadamard_transform")
  module.hadamard_transform = lambda x, scale=1.0: x
  sys.modules["fast_hadamard_transform"] = module


def _restore_module(name: str, previous: types.ModuleType | None) -> None:
  if previous is None:
    sys.modules.pop(name, None)
  else:
    sys.modules[name] = previous


def _causal_mask(batch_size: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
  mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min)
  mask = torch.triu(mask, diagonal=1)
  return mask.view(1, seq_len, seq_len).expand(batch_size, -1, -1)
