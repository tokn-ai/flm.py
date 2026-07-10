from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from flm_llm import ReferenceModel, ReferenceModelConfig
from flm_train.checkpoints import CheckpointState, save_checkpoint
from flm_vllm.export import export_reference_checkpoint, reference_vllm_config
from flm_vllm.importing import import_reference_export
from flm_vllm.reference import (
  FlmReferenceForCausalLM,
  _pad_attention_output,
  _pad_qkv_heads,
  _vllm_weight_name,
)
from flm_vllm.registration import register_flm_models
from flm_vllm.rollout import generate_vllm_rollouts, resolve_export_encoding_name
from safetensors.torch import load_file


def test_reference_vllm_config_uses_experiment_dimensions() -> None:
  assert generate_vllm_rollouts is not None
  config = reference_vllm_config(
    {
      "data": {"seq_len": 1024, "vocab_size": 8192},
      "model": {
        "kind": "reference",
        "d_model": 256,
        "n_layers": 12,
        "n_heads": 16,
        "d_ff": 1024,
      },
    }
  )

  assert config.vocab_size == 8192
  assert config.max_position_embeddings == 1024
  assert config.hidden_size == 256
  assert config.num_hidden_layers == 12
  assert config.num_attention_heads == 16
  assert config.intermediate_size == 1024
  assert config.head_dim == 16


def test_export_reference_checkpoint_writes_vllm_config_and_weights(
  tmp_path: Path,
) -> None:
  run_dir = tmp_path / "runs" / "reference" / "run-123"
  run_dir.mkdir(parents=True)
  (run_dir / "config.json").write_text(
    json.dumps(
      {
        "data": {"seq_len": 16, "vocab_size": 32},
        "model": {
          "kind": "reference",
          "d_model": 8,
          "n_layers": 2,
          "n_heads": 2,
          "d_ff": 24,
        },
      }
    ),
    encoding="utf-8",
  )
  model = ReferenceModel(
    ReferenceModelConfig(
      vocab_size=32,
      max_seq_len=16,
      d_model=8,
      n_layers=2,
      n_heads=2,
      d_ff=24,
    )
  )
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  checkpoint = save_checkpoint(
    checkpoint_dir=run_dir / "checkpoints",
    model=model,
    optimizer=optimizer,
    state=CheckpointState(step=3, tokens_seen=48),
  )

  output_dir = export_reference_checkpoint(
    run_dir=run_dir,
    checkpoint=checkpoint,
  )

  exported_config = json.loads((output_dir / "config.json").read_text("utf-8"))
  assert exported_config["architectures"] == ["FlmReferenceForCausalLM"]
  assert exported_config["model_type"] == "llama"
  assert exported_config["vocab_size"] == 32
  assert exported_config["hidden_size"] == 8
  assert exported_config["num_hidden_layers"] == 2
  assert exported_config["num_attention_heads"] == 2
  assert exported_config["intermediate_size"] == 24
  assert exported_config["tie_word_embeddings"] is True

  manifest = json.loads((output_dir / "flm_vllm_manifest.json").read_text("utf-8"))
  assert manifest["format"] == "flm-vllm-reference-export-v2"
  assert manifest["weight_file"] == "model.safetensors"
  assert manifest["weight_format"] == "safetensors"

  weights = load_file(output_dir / "model.safetensors", device="cpu")
  assert weights["token_embedding.weight"].shape == (32, 8)
  assert weights["blocks.0.attn.qkv.weight"].shape == (24, 8)
  assert weights["blocks.1.ffn.up.weight"].shape == (48, 8)
  assert "lm_head.weight" in weights
  assert not (output_dir / "pytorch_model.bin").exists()
  assert (output_dir / "flm_vllm_manifest.json").is_file()


def test_import_reference_export_round_trips_exported_weights(tmp_path: Path) -> None:
  run_dir = tmp_path / "runs" / "reference" / "run-123"
  run_dir.mkdir(parents=True)
  (run_dir / "config.json").write_text(
    json.dumps(
      {
        "data": {"seq_len": 16, "vocab_size": 32},
        "model": {
          "kind": "reference",
          "d_model": 8,
          "n_layers": 2,
          "n_heads": 2,
          "d_ff": 24,
        },
      }
    ),
    encoding="utf-8",
  )
  model = ReferenceModel(
    ReferenceModelConfig(
      vocab_size=32,
      max_seq_len=16,
      d_model=8,
      n_layers=2,
      n_heads=2,
      d_ff=24,
    )
  )
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  checkpoint = save_checkpoint(
    checkpoint_dir=run_dir / "checkpoints",
    model=model,
    optimizer=optimizer,
    state=CheckpointState(step=3, tokens_seen=48),
  )

  output_dir = export_reference_checkpoint(
    run_dir=run_dir,
    checkpoint=checkpoint,
  )
  imported = import_reference_export(output_dir)
  input_ids = torch.tensor([[1, 2, 3, 4]])

  expected, _ = model(input_ids)
  actual, _ = imported.model(input_ids)

  assert imported.weight_path == output_dir / "model.safetensors"
  assert imported.model.config.vocab_size == 32
  assert imported.model.config.max_seq_len == 16
  torch.testing.assert_close(actual, expected)


def test_import_reference_export_supports_legacy_bin_weight_file(
  tmp_path: Path,
) -> None:
  model_dir = tmp_path / "model"
  model_dir.mkdir()
  config = {
    "architectures": ["FlmReferenceForCausalLM"],
    "attention_bias": False,
    "hidden_size": 8,
    "intermediate_size": 24,
    "max_position_embeddings": 16,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000.0,
    "tie_word_embeddings": True,
    "vocab_size": 32,
  }
  (model_dir / "config.json").write_text(
    json.dumps(config),
    encoding="utf-8",
  )
  model = ReferenceModel(
    ReferenceModelConfig(
      vocab_size=32,
      max_seq_len=16,
      d_model=8,
      n_layers=2,
      n_heads=2,
      d_ff=24,
    )
  )
  torch.save(model.state_dict(), model_dir / "pytorch_model.bin")

  imported = import_reference_export(model_dir)

  assert imported.weight_path == model_dir / "pytorch_model.bin"
  assert imported.model.config.vocab_size == 32


def test_reference_vllm_adapter_exposes_input_embeddings() -> None:
  adapter = FlmReferenceForCausalLM(vllm_config=_vllm_config())
  input_ids = torch.tensor([1, 2, 3])

  embeddings = adapter.embed_input_ids(input_ids)

  assert embeddings.shape == (3, 8)


def test_reference_vllm_adapter_reports_loaded_vllm_parameter_names() -> None:
  adapter = FlmReferenceForCausalLM(vllm_config=_vllm_config())
  state = ReferenceModel(
    ReferenceModelConfig(
      vocab_size=32,
      max_seq_len=16,
      d_model=8,
      n_layers=2,
      n_heads=2,
      d_ff=24,
    )
  ).state_dict()

  loaded = adapter.load_weights(state.items())

  assert "model.token_embedding.weight" in loaded
  assert "model.blocks.0.attn.qkv.weight" in loaded
  assert "token_embedding.weight" not in loaded


def _vllm_config():
  return SimpleNamespace(
    model_config=SimpleNamespace(
      hf_config=SimpleNamespace(
        attention_bias=False,
        hidden_size=8,
        intermediate_size=24,
        max_position_embeddings=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        vocab_size=32,
      )
    )
  )


def test_resolve_export_encoding_name_prefers_explicit_value(tmp_path: Path) -> None:
  model_dir = tmp_path / "model"
  model_dir.mkdir()
  (model_dir / "flm_tokenizer.json").write_text(
    json.dumps({"encoding_name": "unitoken:/tmp/tokenizer"}),
    encoding="utf-8",
  )

  assert (
    resolve_export_encoding_name(
      model_dir=model_dir,
      encoding_name="gpt2",
    )
    == "gpt2"
  )


def test_resolve_export_encoding_name_uses_export_hint(tmp_path: Path) -> None:
  model_dir = tmp_path / "model"
  model_dir.mkdir()
  (model_dir / "flm_tokenizer.json").write_text(
    json.dumps({"encoding_name": "unitoken:/tmp/tokenizer"}),
    encoding="utf-8",
  )

  assert (
    resolve_export_encoding_name(
      model_dir=model_dir,
      encoding_name=None,
    )
    == "unitoken:/tmp/tokenizer"
  )


def test_generate_vllm_rollouts_configures_cpu_options(
  tmp_path: Path,
  monkeypatch,
) -> None:
  engine_options = {}
  monkeypatch.delenv("VLLM_CPU_KVCACHE_SPACE", raising=False)
  monkeypatch.delenv("VLLM_CPU_OMP_THREADS_BIND", raising=False)

  class FakeModelRegistry:
    @staticmethod
    def get_supported_archs():
      return []

    @staticmethod
    def register_model(name, model) -> None:
      del name, model

  class FakeLLM:
    def __init__(self, **kwargs) -> None:
      engine_options.update(kwargs)

    def generate(self, requests, sampling):
      del requests, sampling
      return []

  class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
      del kwargs

  fake_vllm = ModuleType("vllm")
  fake_vllm.LLM = FakeLLM
  fake_vllm.ModelRegistry = FakeModelRegistry
  fake_vllm.SamplingParams = FakeSamplingParams
  monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
  monkeypatch.setattr(
    "flm_vllm.rollout.get_tokenizer",
    lambda _: SimpleNamespace(encode_ordinary=lambda value: [len(value)]),
  )

  batch = generate_vllm_rollouts(
    model_dir=tmp_path,
    encoding_name="test",
    prompts=(),
    max_new_tokens=4,
    dtype="bfloat16",
    cpu_kvcache_space=4,
    cpu_omp_threads_bind="0-3",
    enforce_eager=True,
    max_num_batched_tokens=128,
  )

  assert batch.samples == ()
  assert engine_options == {
    "model": str(tmp_path),
    "tokenizer": None,
    "skip_tokenizer_init": True,
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "enforce_eager": True,
    "max_num_batched_tokens": 128,
  }
  assert os.environ["VLLM_CPU_KVCACHE_SPACE"] == "4"
  assert os.environ["VLLM_CPU_OMP_THREADS_BIND"] == "0-3"


def test_register_flm_models_is_idempotent_and_lazy(monkeypatch) -> None:
  registrations = []
  supported = set()

  class FakeModelRegistry:
    @staticmethod
    def get_supported_archs():
      return supported

    @staticmethod
    def register_model(name, model) -> None:
      registrations.append((name, model))
      supported.add(name)

  fake_vllm = ModuleType("vllm")
  fake_vllm.ModelRegistry = FakeModelRegistry
  monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

  register_flm_models()
  register_flm_models()

  assert registrations == [
    ("FlmReferenceForCausalLM", "flm_vllm.reference:FlmReferenceForCausalLM")
  ]


def test_reference_vllm_weight_names_match_native_llama_layers() -> None:
  assert _vllm_weight_name("token_embedding.weight") == "model.embed_tokens.weight"
  assert _vllm_weight_name("blocks.3.attn_norm.weight") == (
    "model.layers.3.input_layernorm.weight"
  )
  assert _vllm_weight_name("blocks.3.attn.qkv.weight") == (
    "model.layers.3.self_attn.qkv_proj.weight"
  )
  assert _vllm_weight_name("blocks.3.attn.out.weight") == (
    "model.layers.3.self_attn.o_proj.weight"
  )
  assert _vllm_weight_name("blocks.3.ffn_norm.weight") == (
    "model.layers.3.post_attention_layernorm.weight"
  )
  assert _vllm_weight_name("blocks.3.ffn.up.weight") == (
    "model.layers.3.mlp.gate_up_proj.weight"
  )
  assert _vllm_weight_name("blocks.3.ffn.down.weight") == (
    "model.layers.3.mlp.down_proj.weight"
  )
  assert _vllm_weight_name("norm.weight") == "model.norm.weight"
  assert _vllm_weight_name("lm_head.weight") == "lm_head.weight"


def test_reference_vllm_cpu_head_padding_preserves_logical_components() -> None:
  qkv = torch.arange(3 * 2 * 4 * 3).reshape(3 * 2 * 4, 3)
  padded_qkv = _pad_qkv_heads(
    qkv,
    n_heads=2,
    logical_head_dim=4,
    physical_head_dim=8,
  ).reshape(3, 2, 8, 3)
  source_qkv = qkv.reshape(3, 2, 4, 3)

  torch.testing.assert_close(padded_qkv[:2, :, 0:4:2], source_qkv[:2, :, :2])
  torch.testing.assert_close(padded_qkv[:2, :, 4:8:2], source_qkv[:2, :, 2:])
  torch.testing.assert_close(padded_qkv[2, :, :4], source_qkv[2])
  assert torch.count_nonzero(padded_qkv[:2, :, 1:4:2]) == 0
  assert torch.count_nonzero(padded_qkv[:2, :, 5:8:2]) == 0
  assert torch.count_nonzero(padded_qkv[2, :, 4:]) == 0

  output = torch.arange(5 * 2 * 4).reshape(5, 2 * 4)
  padded_output = _pad_attention_output(
    output,
    n_heads=2,
    logical_head_dim=4,
    physical_head_dim=8,
  ).reshape(5, 2, 8)
  torch.testing.assert_close(padded_output[:, :, :4], output.reshape(5, 2, 4))
  assert torch.count_nonzero(padded_output[:, :, 4:]) == 0


def test_reference_vllm_config_rejects_non_reference_model() -> None:
  try:
    reference_vllm_config(
      {
        "data": {"seq_len": 1024, "vocab_size": 8192},
        "model": {"kind": "ds_tiny"},
      }
    )
  except ValueError as exc:
    assert "model.kind='reference'" in str(exc)
  else:
    raise AssertionError("expected non-reference model to be rejected")
