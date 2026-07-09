from pathlib import Path

import numpy as np
import pytest
import torch
from flm_modules.attentions.mla import DeepSeekMLA
from flm_modules.teaching.mla import DeepSeekMLA as TeachingDeepSeekMLA
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
  DeepseekV3Attention,
  DeepseekV3RotaryEmbedding,
)

MLABackend = str


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_mla_preserves_model_dimension(
  random_input,
  backend: MLABackend,
) -> None:
  layer = _mla(backend)
  x = random_input(2, 5, 8)

  y = layer(x)

  assert y.shape == x.shape


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_mla_supports_q_lora_variant(
  random_input,
  backend: MLABackend,
) -> None:
  layer = _mla(backend, q_lora_rank=5)
  x = random_input(2, 5, 8)

  y = layer(x)

  assert y.shape == x.shape


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_mla_matches_golden(
  backend: MLABackend,
) -> None:
  golden = np.load(Path(__file__).parent / "golden" / "mla.npz")
  layer = _mla(backend, bias=False, rope_layout="llama")
  _load_transformers_npz(layer, golden, q_lora_rank=None)
  x = torch.from_numpy(golden["hidden_states"])
  attention_mask = torch.from_numpy(golden["attention_mask"])
  expected = torch.from_numpy(golden["output"])

  torch.testing.assert_close(layer(x, attention_mask=attention_mask), expected)


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_mla_matches_transformers_attention(
  random_input,
  backend: MLABackend,
) -> None:
  config = _deepseek_config(q_lora_rank=None)
  reference = DeepseekV3Attention(config, layer_idx=0)
  layer = _mla(backend, q_lora_rank=None, bias=False, rope_layout="llama")
  x = random_input(2, 5, 8)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
  rope = DeepseekV3RotaryEmbedding(config)
  position_embeddings = rope(x, position_ids)
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)

  _load_transformers_module(layer, reference, q_lora_rank=None)

  expected, _ = reference(
    x,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
  )

  torch.testing.assert_close(layer(x, attention_mask=attention_mask), expected)


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_mla_q_lora_matches_transformers_attention(
  random_input,
  backend: MLABackend,
) -> None:
  config = _deepseek_config(q_lora_rank=5)
  reference = DeepseekV3Attention(config, layer_idx=0)
  layer = _mla(backend, q_lora_rank=5, bias=False, rope_layout="llama")
  x = random_input(2, 5, 8)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
  rope = DeepseekV3RotaryEmbedding(config)
  position_embeddings = rope(x, position_ids)
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)

  _load_transformers_module(layer, reference, q_lora_rank=5)

  expected, _ = reference(
    x,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
  )

  torch.testing.assert_close(layer(x, attention_mask=attention_mask), expected)


@pytest.mark.parametrize("q_lora_rank", [None, 5])
def test_deepseek_mla_matches_attention_implementation(
  random_input,
  q_lora_rank: int | None,
) -> None:
  reference = _mla("torch", q_lora_rank=q_lora_rank, bias=False, rope_layout="llama")
  layer = _mla("teaching", q_lora_rank=q_lora_rank, bias=False, rope_layout="llama")
  x = random_input(2, 5, 8)
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)

  _load_transformers_module(layer, reference, q_lora_rank=q_lora_rank)

  expected = reference(x, attention_mask=attention_mask)
  actual = layer(x, attention_mask=attention_mask)

  torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_mla_backpropagates(
  random_input,
  backend: MLABackend,
) -> None:
  layer = _mla(backend, q_lora_rank=5)
  x = random_input(2, 5, 8).requires_grad_()

  layer(x).square().mean().backward()

  assert x.grad is not None
  assert layer.o_proj.weight.grad is not None
  if backend == "teaching":
    assert layer.q_d_proj.weight.grad is not None
    assert layer.kv_d_proj.weight.grad is not None
  else:
    assert layer.q_a_proj.weight.grad is not None
    assert layer.kv_a_proj_with_mqa.weight.grad is not None


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_mla_rejects_invalid_rope_dimension(
  backend: MLABackend,
) -> None:
  with pytest.raises(ValueError, match="positive even"):
    _mla(backend, qk_rope_head_dim=3)


def _mla(
  backend: MLABackend,
  *,
  d_model: int = 8,
  n_heads: int = 2,
  kv_lora_rank: int = 6,
  q_lora_rank: int | None = None,
  qk_nope_head_dim: int = 4,
  qk_rope_head_dim: int = 4,
  v_head_dim: int = 4,
  bias: bool = False,
  rope_layout: str = "llama",
):
  cls = {
    "teaching": TeachingDeepSeekMLA,
    "torch": DeepSeekMLA,
  }[backend]
  return cls(
    d_model=d_model,
    n_heads=n_heads,
    kv_lora_rank=kv_lora_rank,
    q_lora_rank=q_lora_rank,
    qk_nope_head_dim=qk_nope_head_dim,
    qk_rope_head_dim=qk_rope_head_dim,
    v_head_dim=v_head_dim,
    bias=bias,
    rope_layout=rope_layout,
    backend="torch",
  )


def _load_transformers_npz(
  layer: torch.nn.Module,
  golden: np.lib.npyio.NpzFile,
  *,
  q_lora_rank: int | None,
) -> None:
  weights = {
    key.removeprefix("transformers."): torch.from_numpy(golden[key])
    for key in golden.files
    if key.startswith("transformers.")
  }
  _load_transformers_weights(layer, weights, q_lora_rank=q_lora_rank)


def _load_transformers_module(
  layer: torch.nn.Module,
  reference: torch.nn.Module,
  *,
  q_lora_rank: int | None,
) -> None:
  _load_transformers_weights(
    layer,
    dict(reference.named_parameters()),
    q_lora_rank=q_lora_rank,
  )


def _load_transformers_weights(
  layer: torch.nn.Module,
  weights: dict[str, torch.Tensor],
  *,
  q_lora_rank: int | None,
) -> None:
  with torch.no_grad():
    if isinstance(layer, TeachingDeepSeekMLA):
      layer.set_weights_from_transformers(
        q_proj=weights["q_proj.weight"] if q_lora_rank is None else None,
        q_a_proj=None if q_lora_rank is None else weights["q_a_proj.weight"],
        q_a_layernorm=None if q_lora_rank is None else weights["q_a_layernorm.weight"],
        q_b_proj=None if q_lora_rank is None else weights["q_b_proj.weight"],
        kv_a_proj_with_mqa=weights["kv_a_proj_with_mqa.weight"],
        kv_a_layernorm=weights["kv_a_layernorm.weight"],
        kv_b_proj=weights["kv_b_proj.weight"],
        o_proj=weights["o_proj.weight"],
      )
      return

    layer.load_state_dict(weights)


def _deepseek_config(q_lora_rank: int | None) -> DeepseekV3Config:
  return DeepseekV3Config(
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=2,
    q_lora_rank=q_lora_rank,
    kv_lora_rank=6,
    qk_rope_head_dim=4,
    qk_nope_head_dim=4,
    qk_head_dim=8,
    v_head_dim=4,
    rope_interleave=False,
    attention_bias=False,
    attention_dropout=0.0,
  )


def _causal_mask(batch_size: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
  mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min)
  mask = torch.triu(mask, diagonal=1)
  return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1)
