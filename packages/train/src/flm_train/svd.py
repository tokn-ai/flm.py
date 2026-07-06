"""Post-checkpoint SVD diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flm_train.sinks.base import Scalar


def checkpoint_ffn_down_svd_metrics(checkpoint_path: Path) -> dict[str, Scalar]:
  metadata = json.loads(
    (checkpoint_path / "model_state.json").read_text(encoding="utf-8")
  )
  with np.load(checkpoint_path / "model.npz") as arrays:
    names = sorted(_ffn_down_weight_names(arrays.files), key=_layer_index)
    if not names:
      return {}

    selected = _selected_layer_names(names)
    metrics: dict[str, Scalar] = {}
    for label, name in selected.items():
      singular_values = torch.linalg.svdvals(_load_tensor(name, arrays, metadata)).cpu()
      metrics.update(
        {
          f"svd/ffn_down/{label}/layer": _layer_index(name),
          f"svd/ffn_down/{label}/effective_rank": _effective_rank(singular_values),
          f"svd/ffn_down/{label}/stable_rank": _stable_rank(singular_values),
          f"svd/ffn_down/{label}/r90": _energy_rank(singular_values, 0.90),
          f"svd/ffn_down/{label}/r95": _energy_rank(singular_values, 0.95),
          f"svd/ffn_down/{label}/r99": _energy_rank(singular_values, 0.99),
          f"svd/ffn_down/{label}/energy_sum": float(
            singular_values.square().sum().item()
          ),
        }
      )
  return metrics


def _ffn_down_weight_names(names: list[str]) -> list[str]:
  return [
    name
    for name in names
    if name.startswith("blocks.")
    and name.endswith(".ffn.down.weight")
  ]


def _selected_layer_names(names: list[str]) -> dict[str, str]:
  last_index = len(names) - 1
  return {
    "first": names[0],
    "quarter": names[round(last_index * 0.25)],
    "last": names[last_index],
  }


def _layer_index(name: str) -> int:
  return int(name.split(".")[1])


def _load_tensor(
  name: str,
  arrays: np.lib.npyio.NpzFile,
  metadata: dict[str, object],
) -> torch.Tensor:
  tensor_metadata = metadata[name]["__tensor__"]
  tensor = torch.from_numpy(arrays[name])
  if tensor_metadata["dtype"] == "torch.bfloat16":
    tensor = tensor.view(torch.bfloat16)
  return tensor.float()


def _effective_rank(singular_values: torch.Tensor) -> float:
  squared = singular_values.square()
  energy = squared / squared.sum()
  entropy = -(
    energy * torch.log(energy.clamp_min(torch.finfo(energy.dtype).tiny))
  ).sum()
  return float(torch.exp(entropy).item())


def _stable_rank(singular_values: torch.Tensor) -> float:
  squared = singular_values.square()
  return float((squared.sum() / squared.max()).item())


def _energy_rank(singular_values: torch.Tensor, target: float) -> int:
  squared = singular_values.square()
  cumulative = torch.cumsum(squared / squared.sum(), dim=0)
  return int(torch.searchsorted(cumulative, torch.tensor(target)).item()) + 1
