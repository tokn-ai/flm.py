import importlib.util
import sys
import types
from pathlib import Path

from flm_llm import DeepSeekV4, DeepSeekV4Config
from flm_modules import (
  DeepSeekV4AttentionKind,
  DeepSeekV4CSACompressor,
  DeepSeekV4HCACompressor,
)
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HF_INFERENCE_PATH = _REPO_ROOT / "huggingface_models" / "deepseek_v4_inference.py"


def test_deepseek_v4_model_topology_matches_transformers_and_hf_inference() -> None:
  layer_types = (
    DeepSeekV4AttentionKind.HEAVILY_COMPRESSED,
    DeepSeekV4AttentionKind.COMPRESSED_SPARSE,
    DeepSeekV4AttentionKind.SLIDING,
  )
  local = DeepSeekV4(_local_config(layer_types))
  transformers = DeepseekV4Model(
    _transformers_config([kind.value for kind in layer_types])
  )
  hf_module = _load_hf_inference_model()
  hf = hf_module.Transformer(_hf_args())

  assert len(local.blocks) == len(transformers.layers) == len(hf.layers) == 3
  assert local.config.hc_mult == transformers.config.hc_mult == hf.hc_mult == 2

  assert [block.attn.layer_type for block in local.blocks] == list(layer_types)
  assert [layer.self_attn.layer_type for layer in transformers.layers] == [
    kind.value for kind in layer_types
  ]
  assert [_hf_ratio_to_kind(layer.attn.compress_ratio) for layer in hf.layers] == list(
    layer_types
  )

  assert isinstance(local.blocks[0].attn.compressor, DeepSeekV4HCACompressor)
  assert isinstance(local.blocks[1].attn.compressor, DeepSeekV4CSACompressor)
  assert local.blocks[2].attn.compressor is None


def _local_config(
  layer_types: tuple[DeepSeekV4AttentionKind, ...],
) -> DeepSeekV4Config:
  return DeepSeekV4Config(
    vocab_size=32,
    max_seq_len=8,
    d_model=16,
    n_layers=3,
    n_heads=2,
    head_dim=8,
    q_lora_rank=8,
    rope_head_dim=8,
    o_lora_rank=4,
    o_groups=2,
    attention_layer_types=layer_types,
    compress_rate_csa=4,
    compress_rate_hca=128,
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    moe_d_ff=16,
    n_routed_experts=4,
    n_shared_experts=1,
    n_experts_per_token=2,
    hc_mult=2,
    hc_sinkhorn_iters=3,
  )


def _transformers_config(layer_types: list[str]) -> DeepseekV4Config:
  return DeepseekV4Config(
    vocab_size=32,
    hidden_size=16,
    num_hidden_layers=3,
    num_attention_heads=2,
    head_dim=8,
    q_lora_rank=8,
    o_lora_rank=4,
    o_groups=2,
    moe_intermediate_size=16,
    n_routed_experts=4,
    num_local_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
    hc_mult=2,
    hc_sinkhorn_iters=3,
    layer_types=layer_types,
    mlp_layer_types=["moe", "moe", "moe"],
    compress_rates={
      "compressed_sparse_attention": 4,
      "heavily_compressed_attention": 128,
    },
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    rope_parameters={
      "main": {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
      },
      "compress": {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
      },
    },
    _attn_implementation="eager",
  )


def _hf_args():
  hf_module = _load_hf_inference_model()
  return hf_module.ModelArgs(
    max_batch_size=2,
    max_seq_len=8,
    dtype="bf16",
    scale_fmt=None,
    expert_dtype=None,
    scale_dtype="fp32",
    vocab_size=32,
    dim=16,
    moe_inter_dim=16,
    n_layers=3,
    n_hash_layers=0,
    n_mtp_layers=0,
    n_heads=2,
    n_routed_experts=4,
    n_shared_experts=1,
    n_activated_experts=2,
    q_lora_rank=8,
    head_dim=8,
    rope_head_dim=8,
    o_groups=2,
    o_lora_rank=4,
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    hc_mult=2,
    hc_sinkhorn_iters=3,
    compress_ratios=(128, 4, 0),
  )


def _hf_ratio_to_kind(ratio: int) -> DeepSeekV4AttentionKind:
  if ratio == 0:
    return DeepSeekV4AttentionKind.SLIDING
  if ratio == 4:
    return DeepSeekV4AttentionKind.COMPRESSED_SPARSE
  return DeepSeekV4AttentionKind.HEAVILY_COMPRESSED


def _load_hf_inference_model():
  _install_kernel_stub()
  module_name = "hf_deepseek_v4_flash_inference_model_for_llm_tests"
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
