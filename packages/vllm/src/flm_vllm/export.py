"""Export FLM checkpoints into a vLLM-loadable directory."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flm_train.checkpoints import _decode_state as decode_checkpoint_state
from flm_train.checkpoints import resolve_checkpoint_path
from safetensors.torch import save_file

WEIGHT_FILE = "model.safetensors"


@dataclass(frozen=True)
class FlmReferenceVllmConfig:
  vocab_size: int
  max_position_embeddings: int
  hidden_size: int
  num_hidden_layers: int
  num_attention_heads: int
  intermediate_size: int
  rope_theta: float
  rms_norm_eps: float
  attention_bias: bool
  tie_word_embeddings: bool = True

  @property
  def head_dim(self) -> int:
    return self.hidden_size // self.num_attention_heads


def export_reference_checkpoint(
  *,
  run_dir: Path,
  checkpoint: Path | str = "latest",
  output_dir: Path | None = None,
) -> Path:
  """Export a reference-model FLM checkpoint for vLLM rollout."""
  run_dir = Path(run_dir)
  config_path = run_dir / "config.json"
  if not config_path.is_file():
    raise FileNotFoundError(config_path)

  experiment_config = _read_json(config_path)
  vllm_config = reference_vllm_config(experiment_config)
  checkpoint_path = _checkpoint_path(run_dir=run_dir, checkpoint=checkpoint)
  model_state = _load_model_state(checkpoint_path)

  output_dir = output_dir or (run_dir / "vllm" / checkpoint_path.name)
  output_dir.mkdir(parents=True, exist_ok=True)
  save_file(_safetensors_state(model_state), output_dir / WEIGHT_FILE)
  _write_json(output_dir / "config.json", _hf_config_payload(vllm_config))
  _write_json(
    output_dir / "flm_vllm_manifest.json",
    {
      "format": "flm-vllm-reference-export-v2",
      "source_run_dir": str(run_dir),
      "source_checkpoint": str(checkpoint_path),
      "weight_file": WEIGHT_FILE,
      "weight_format": "safetensors",
      "config": asdict(vllm_config),
    },
  )
  _copy_tokenizer_hint(run_dir=run_dir, output_dir=output_dir)
  return output_dir


def reference_vllm_config(experiment_config: dict[str, Any]) -> FlmReferenceVllmConfig:
  model = _mapping(experiment_config.get("model"), name="model")
  data = _mapping(experiment_config.get("data"), name="data")
  if model.get("kind") != "reference":
    raise ValueError("flm-vllm currently supports only model.kind='reference'")

  d_model = int(model.get("d_model", 768))
  n_heads = int(model.get("n_heads", 12))
  if d_model % n_heads != 0:
    raise ValueError("model.d_model must be divisible by model.n_heads")
  d_ff = model.get("d_ff")
  intermediate_size = int(d_ff) if d_ff is not None else int(8 * d_model / 3)
  vocab_size = _vocab_size(experiment_config)
  return FlmReferenceVllmConfig(
    vocab_size=vocab_size,
    max_position_embeddings=int(data.get("seq_len", 2048)),
    hidden_size=d_model,
    num_hidden_layers=int(model.get("n_layers", 12)),
    num_attention_heads=n_heads,
    intermediate_size=intermediate_size,
    rope_theta=float(model.get("rope_base", 10_000.0)),
    rms_norm_eps=float(model.get("norm_eps", 1e-6)),
    attention_bias=bool(model.get("bias", False)),
  )


def _hf_config_payload(config: FlmReferenceVllmConfig) -> dict[str, Any]:
  payload = {
    "architectures": ["FlmReferenceForCausalLM"],
    # vLLM still asks Hugging Face tooling to parse config.json. Reusing the
    # Llama config envelope gives us standard dimensional fields while the
    # architecture entry selects the FLM adapter registered by flm-vllm.
    "model_type": "llama",
    "vocab_size": config.vocab_size,
    "max_position_embeddings": config.max_position_embeddings,
    "hidden_size": config.hidden_size,
    "num_hidden_layers": config.num_hidden_layers,
    "num_attention_heads": config.num_attention_heads,
    "num_key_value_heads": config.num_attention_heads,
    "head_dim": config.head_dim,
    "intermediate_size": config.intermediate_size,
    "hidden_act": "silu",
    "rope_theta": config.rope_theta,
    "rms_norm_eps": config.rms_norm_eps,
    "attention_bias": config.attention_bias,
    "mlp_bias": config.attention_bias,
    "tie_word_embeddings": config.tie_word_embeddings,
    "bos_token_id": 0,
    "eos_token_id": 0,
  }
  return payload


def _load_model_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
  metadata = _read_json(checkpoint_path / "model_state.json")
  with np.load(checkpoint_path / "model.npz") as arrays:
    return decode_checkpoint_state(
      metadata=metadata,
      arrays=arrays,
      map_location="cpu",
    )


def _safetensors_state(
  state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  # Safetensors intentionally rejects shared storage. FLM ties embedding and
  # lm_head weights, so clone tensors into an explicit flat serving payload.
  return {
    name: tensor.detach().cpu().contiguous().clone()
    for name, tensor in state.items()
  }


def _checkpoint_path(*, run_dir: Path, checkpoint: Path | str) -> Path:
  path = Path(checkpoint)
  if not path.is_absolute():
    path = run_dir / "checkpoints" / path
  return resolve_checkpoint_path(path)


def _copy_tokenizer_hint(*, run_dir: Path, output_dir: Path) -> None:
  config = _read_json(run_dir / "config.json")
  encoding_name = _mapping(config.get("data"), name="data").get("encoding_name")
  if isinstance(encoding_name, str):
    _write_json(output_dir / "flm_tokenizer.json", {"encoding_name": encoding_name})
  tokenizer_dir = _tokenizer_dir_from_encoding(encoding_name)
  if tokenizer_dir is not None and tokenizer_dir.is_dir():
    target = output_dir / "flm_tokenizer"
    if target.exists():
      shutil.rmtree(target)
    shutil.copytree(tokenizer_dir, target)


def _tokenizer_dir_from_encoding(encoding_name: object) -> Path | None:
  if not isinstance(encoding_name, str):
    return None
  prefix = "unitoken:"
  if not encoding_name.startswith(prefix):
    return None
  path = Path(encoding_name[len(prefix) :])
  return path if path.is_absolute() else None


def _vocab_size(experiment_config: dict[str, Any]) -> int:
  data = _mapping(experiment_config.get("data"), name="data")
  for key in ("vocab_size", "n_vocab"):
    value = data.get(key)
    if value is not None:
      return int(value)
  encoding_name = data.get("encoding_name")
  if not isinstance(encoding_name, str):
    raise ValueError("data.encoding_name is required to infer vocab size")
  from flm_datasets import get_tokenizer

  return int(get_tokenizer(encoding_name).n_vocab)


def _mapping(value: object, *, name: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise ValueError(f"{name} must be a mapping")
  return value


def _read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
