from pathlib import Path

import numpy as np
import pytest
import torch
from flm_modules.attentions.dsa import DeepSeekDSA, DeepSeekDSAIndexer
from flm_modules.teaching.dsa import DeepSeekDSA as TeachingDeepSeekDSA
from flm_modules.teaching.dsa import DeepSeekDSAIndexer as TeachingDeepSeekDSAIndexer
from transformers import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
  DeepseekV32Attention,
  DeepseekV32Indexer,
  DeepseekV32RotaryEmbedding,
)

DSABackend = str


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_dsa_indexer_matches_golden(
  backend: DSABackend,
) -> None:
  golden = np.load(Path(__file__).parent / "golden" / "dsa.npz")
  layer = _dsa_indexer(backend)
  _load_indexer_weights(layer, golden)
  x = torch.from_numpy(golden["indexer.hidden_states"])
  q_residual = torch.from_numpy(golden["indexer.q_residual"])
  attention_mask = torch.from_numpy(golden["indexer.attention_mask"])
  positions = torch.from_numpy(golden["indexer.position_ids"])
  expected = torch.from_numpy(golden["indexer.output"])

  actual = layer(
    x,
    q_residual,
    attention_mask=attention_mask,
    positions=positions,
  )
  print(actual, expected)

  torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_dsa_matches_golden(
  backend: DSABackend,
) -> None:
  golden = np.load(Path(__file__).parent / "golden" / "dsa.npz")
  layer = _dsa(backend)
  _load_dsa_weights(layer, golden)
  x = torch.from_numpy(golden["attention.hidden_states"])
  attention_mask = torch.from_numpy(golden["attention.attention_mask"])
  positions = torch.from_numpy(golden["attention.position_ids"])
  expected = torch.from_numpy(golden["attention.output"])

  actual = layer(x, attention_mask=attention_mask, positions=positions)

  torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_dsa_indexer_matches_transformers(
  random_input,
  backend: DSABackend,
) -> None:
  config = _deepseek_v32_config()
  reference = DeepseekV32Indexer(config, layer_idx=0)
  layer = _dsa_indexer(backend)
  x = random_input(2, 5, 8)
  q_residual = random_input(2, 5, 5)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
  position_embeddings = DeepseekV32RotaryEmbedding(config)(x, position_ids)
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)[:, 0]

  _load_indexer_module(layer, reference)

  expected = reference(
    x,
    q_residual,
    position_embeddings,
    attention_mask,
    position_ids,
  )

  torch.testing.assert_close(
    layer(x, q_residual, attention_mask=attention_mask, positions=position_ids),
    expected,
  )


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_dsa_matches_transformers_attention(
  random_input,
  backend: DSABackend,
) -> None:
  config = _deepseek_v32_config()
  reference = DeepseekV32Attention(config, layer_idx=0)
  layer = _dsa(backend)
  x = random_input(2, 5, 8)
  position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
  position_embeddings = DeepseekV32RotaryEmbedding(config)(x, position_ids)
  attention_mask = _causal_mask(batch_size=2, seq_len=5, dtype=x.dtype)

  _load_dsa_module(layer, reference)

  expected, _ = reference(
    x,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
    position_ids=position_ids,
  )

  torch.testing.assert_close(
    layer(x, attention_mask=attention_mask, positions=position_ids),
    expected,
  )


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_dsa_backpropagates(
  random_input,
  backend: DSABackend,
) -> None:
  layer = _dsa(backend)
  x = random_input(2, 5, 8).requires_grad_()

  layer(x).square().mean().backward()

  assert x.grad is not None
  assert layer.q_a_proj.weight.grad is not None
  assert layer.kv_a_proj_with_mqa.weight.grad is not None
  assert layer.o_proj.weight.grad is not None


@pytest.mark.parametrize("backend", ["teaching", "torch"])
def test_deepseek_dsa_rejects_invalid_index_rope_dimension(
  backend: DSABackend,
) -> None:
  if backend == "teaching":
    with pytest.raises(NotImplementedError, match="DeepSeekDSAIndexer"):
      _dsa_indexer(backend, qk_rope_head_dim=8)
    return

  with pytest.raises(ValueError, match="must not exceed index_head_dim"):
    _dsa_indexer(backend, qk_rope_head_dim=8)


def test_teaching_deepseek_dsa_indexer_init_is_scaffold() -> None:
  with pytest.raises(NotImplementedError, match="DeepSeekDSAIndexer"):
    _dsa_indexer("teaching", skip_unimplemented=False)


def test_teaching_deepseek_dsa_init_is_scaffold() -> None:
  with pytest.raises(NotImplementedError, match="DeepSeekDSA"):
    _dsa("teaching", skip_unimplemented=False)


def _dsa_indexer(
  backend: DSABackend,
  *,
  d_model: int = 8,
  q_lora_rank: int = 5,
  qk_rope_head_dim: int = 4,
  index_n_heads: int = 3,
  index_head_dim: int = 6,
  index_topk: int = 3,
  rope_base: float = 10_000.0,
  skip_unimplemented: bool = True,
):
  cls = {
    "teaching": TeachingDeepSeekDSAIndexer,
    "torch": DeepSeekDSAIndexer,
  }[backend]
  try:
    return cls(
      d_model=d_model,
      q_lora_rank=q_lora_rank,
      qk_rope_head_dim=qk_rope_head_dim,
      index_n_heads=index_n_heads,
      index_head_dim=index_head_dim,
      index_topk=index_topk,
      rope_base=rope_base,
    )
  except NotImplementedError as exc:
    if skip_unimplemented:
      pytest.skip(str(exc))
    raise


def _dsa(
  backend: DSABackend,
  *,
  d_model: int = 8,
  n_heads: int = 2,
  kv_lora_rank: int = 6,
  q_lora_rank: int = 5,
  qk_nope_head_dim: int = 4,
  qk_rope_head_dim: int = 4,
  v_head_dim: int = 4,
  index_n_heads: int = 3,
  index_head_dim: int = 6,
  index_topk: int = 3,
  bias: bool = False,
  rope_base: float = 10_000.0,
  norm_eps: float = 1e-6,
  skip_unimplemented: bool = True,
):
  cls = {
    "teaching": TeachingDeepSeekDSA,
    "torch": DeepSeekDSA,
  }[backend]
  try:
    return cls(
      d_model=d_model,
      n_heads=n_heads,
      kv_lora_rank=kv_lora_rank,
      q_lora_rank=q_lora_rank,
      qk_nope_head_dim=qk_nope_head_dim,
      qk_rope_head_dim=qk_rope_head_dim,
      v_head_dim=v_head_dim,
      index_n_heads=index_n_heads,
      index_head_dim=index_head_dim,
      index_topk=index_topk,
      bias=bias,
      rope_base=rope_base,
      norm_eps=norm_eps,
      backend="torch",
    )
  except NotImplementedError as exc:
    if skip_unimplemented:
      pytest.skip(str(exc))
    raise


def _load_indexer_weights(
  layer: torch.nn.Module,
  golden: np.lib.npyio.NpzFile,
) -> None:
  weights = {
    key.removeprefix("indexer.transformers."): torch.from_numpy(golden[key])
    for key in golden.files
    if key.startswith("indexer.transformers.")
  }
  _load_indexer_transformers_weights(layer, weights)


def _load_dsa_weights(
  layer: torch.nn.Module,
  golden: np.lib.npyio.NpzFile,
) -> None:
  weights = {
    key.removeprefix("attention.transformers."): torch.from_numpy(golden[key])
    for key in golden.files
    if key.startswith("attention.transformers.")
  }
  _load_dsa_transformers_weights(layer, weights)


def _load_indexer_module(
  layer: torch.nn.Module,
  reference: torch.nn.Module,
) -> None:
  _load_indexer_transformers_weights(layer, reference.state_dict())


def _load_dsa_module(
  layer: torch.nn.Module,
  reference: torch.nn.Module,
) -> None:
  _load_dsa_transformers_weights(layer, reference.state_dict())


def _load_indexer_transformers_weights(
  layer: torch.nn.Module,
  weights: dict[str, torch.Tensor],
) -> None:
  if isinstance(layer, TeachingDeepSeekDSAIndexer):
    layer.set_weights_from_transformers(
      wq_b=weights["wq_b.weight"],
      wk=weights["wk.weight"],
      k_norm_weight=weights["k_norm.weight"],
      k_norm_bias=weights["k_norm.bias"],
      weights_proj=weights["weights_proj.weight"],
    )
    return

  layer.load_state_dict(weights)


def _load_dsa_transformers_weights(
  layer: torch.nn.Module,
  weights: dict[str, torch.Tensor],
) -> None:
  if isinstance(layer, TeachingDeepSeekDSA):
    layer.set_weights_from_transformers(
      q_a_proj=weights["q_a_proj.weight"],
      q_a_layernorm=weights["q_a_layernorm.weight"],
      q_b_proj=weights["q_b_proj.weight"],
      kv_a_proj_with_mqa=weights["kv_a_proj_with_mqa.weight"],
      kv_a_layernorm=weights["kv_a_layernorm.weight"],
      kv_b_proj=weights["kv_b_proj.weight"],
      o_proj=weights["o_proj.weight"],
      indexer_wq_b=weights["indexer.wq_b.weight"],
      indexer_wk=weights["indexer.wk.weight"],
      indexer_k_norm_weight=weights["indexer.k_norm.weight"],
      indexer_k_norm_bias=weights["indexer.k_norm.bias"],
      indexer_weights_proj=weights["indexer.weights_proj.weight"],
    )
    return

  layer.load_state_dict(weights)


def _deepseek_v32_config() -> DeepseekV32Config:
  config = DeepseekV32Config(
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=2,
    q_lora_rank=5,
    kv_lora_rank=6,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
    index_n_heads=3,
    index_head_dim=6,
    index_topk=3,
    attention_bias=False,
    attention_dropout=0.0,
    rope_parameters={
      "rope_type": "default",
      "rope_theta": 10_000.0,
    },
  )
  config._attn_implementation = "eager"
  return config


def _causal_mask(batch_size: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
  mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min)
  mask = torch.triu(mask, diagonal=1)
  return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1)
