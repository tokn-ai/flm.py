from __future__ import annotations

import json
from pathlib import Path

import torch
from flm_llm import ReferenceModel, ReferenceModelConfig
from flm_train.checkpoints import CheckpointState, save_checkpoint
from flm_vllm.export import export_reference_checkpoint, reference_vllm_config
from flm_vllm.rollout import generate_vllm_rollouts


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

  weights = torch.load(
    output_dir / "pytorch_model.bin",
    map_location="cpu",
    weights_only=True,
  )
  assert weights["token_embedding.weight"].shape == (32, 8)
  assert weights["blocks.0.attn.qkv.weight"].shape == (24, 8)
  assert weights["blocks.1.ffn.up.weight"].shape == (48, 8)
  assert "lm_head.weight" in weights
  assert (output_dir / "flm_vllm_manifest.json").is_file()


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
