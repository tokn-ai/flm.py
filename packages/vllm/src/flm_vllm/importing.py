"""Import FLM vLLM exports back into local FLM models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from flm_llm import ReferenceModel, ReferenceModelConfig
from safetensors.torch import load_file

SAFETENSORS_WEIGHT_FILE = "model.safetensors"
LEGACY_BIN_WEIGHT_FILE = "pytorch_model.bin"


@dataclass(frozen=True)
class ImportedReferenceModel:
  model: ReferenceModel
  config: dict[str, Any]
  manifest: dict[str, Any]
  weight_path: Path


def import_reference_export(
  model_dir: Path,
  *,
  map_location: str = "cpu",
) -> ImportedReferenceModel:
  """Load a reference-model vLLM export into the native FLM model.

  This is the local counterpart to the vLLM adapter's ``load_weights`` path. It
  gives tests and tools a way to validate exported model directories without a
  vLLM runtime.
  """
  model_dir = Path(model_dir)
  config = _read_json(model_dir / "config.json")
  manifest = _read_manifest(model_dir)
  weight_path = _resolve_weight_path(model_dir=model_dir, manifest=manifest)
  state = _load_weights(weight_path=weight_path, map_location=map_location)
  _validate_export_config(config)
  _validate_tied_embeddings(state)

  model = ReferenceModel(_reference_config_from_hf_config(config))
  model.load_state_dict(state)
  return ImportedReferenceModel(
    model=model,
    config=config,
    manifest=manifest,
    weight_path=weight_path,
  )


def _reference_config_from_hf_config(config: dict[str, Any]) -> ReferenceModelConfig:
  return ReferenceModelConfig(
    vocab_size=int(config["vocab_size"]),
    max_seq_len=int(config["max_position_embeddings"]),
    d_model=int(config["hidden_size"]),
    n_layers=int(config["num_hidden_layers"]),
    n_heads=int(config["num_attention_heads"]),
    d_ff=int(config["intermediate_size"]),
    bias=bool(config.get("attention_bias", False)),
    rope_base=float(config.get("rope_theta", 10_000.0)),
    norm_eps=float(config.get("rms_norm_eps", 1e-6)),
  )


def _validate_export_config(config: dict[str, Any]) -> None:
  architectures = config.get("architectures")
  if architectures != ["FlmReferenceForCausalLM"]:
    raise ValueError(
      "expected a FlmReferenceForCausalLM export, "
      f"got architectures={architectures!r}"
    )
  if bool(config.get("tie_word_embeddings", True)) is not True:
    raise ValueError("FLM reference exports require tied word embeddings")


def _validate_tied_embeddings(state: dict[str, torch.Tensor]) -> None:
  embedding = state.get("token_embedding.weight")
  head = state.get("lm_head.weight")
  if embedding is None or head is None:
    raise ValueError("export is missing tied embedding/head weights")
  if embedding.shape != head.shape or not torch.equal(embedding, head):
    raise ValueError("exported lm_head.weight must match token_embedding.weight")


def _load_weights(*, weight_path: Path, map_location: str) -> dict[str, torch.Tensor]:
  if not weight_path.is_file():
    raise FileNotFoundError(weight_path)
  if weight_path.suffix == ".safetensors":
    state = load_file(weight_path, device=map_location)
  else:
    state = torch.load(
      weight_path,
      map_location=map_location,
      weights_only=True,
    )
  if not isinstance(state, dict):
    raise ValueError(f"expected state dict in {weight_path}")
  if not all(isinstance(value, torch.Tensor) for value in state.values()):
    raise ValueError(f"expected tensor-only state dict in {weight_path}")
  return state


def _resolve_weight_path(*, model_dir: Path, manifest: dict[str, Any]) -> Path:
  weight_file = manifest.get("weight_file")
  if isinstance(weight_file, str):
    return model_dir / weight_file

  safetensors_path = model_dir / SAFETENSORS_WEIGHT_FILE
  if safetensors_path.is_file():
    return safetensors_path
  return model_dir / LEGACY_BIN_WEIGHT_FILE


def _read_manifest(model_dir: Path) -> dict[str, Any]:
  path = model_dir / "flm_vllm_manifest.json"
  if not path.is_file():
    return {}
  return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))
