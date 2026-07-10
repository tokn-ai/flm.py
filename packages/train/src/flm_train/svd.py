"""Post-checkpoint SVD diagnostics."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flm_train.sinks.base import Scalar


@dataclass(frozen=True)
class _LayerSvdMetrics:
  layer: int
  effective_rank: float
  stable_rank: float
  r90: int
  r95: int
  r99: int
  energy_sum: float


def checkpoint_ffn_down_svd_metrics(checkpoint_path: Path) -> dict[str, Scalar]:
  metadata = json.loads(
    (checkpoint_path / "model_state.json").read_text(encoding="utf-8")
  )
  with np.load(checkpoint_path / "model.npz") as arrays:
    names = sorted(_ffn_down_weight_names(arrays.files), key=_layer_index)
    if not names:
      return {}

    selected = _selected_layer_names(names)
    layer_metrics = {
      name: _svd_metrics(
        layer=_layer_index(name),
        singular_values=torch.linalg.svdvals(
          _load_tensor(name, arrays, metadata)
        ).cpu(),
      )
      for name in names
    }
    layer_label_width = max(
      2,
      len(str(max(metric.layer for metric in layer_metrics.values()))),
    )
    metrics: dict[str, Scalar] = {}
    for name in selected:
      selected_metrics = layer_metrics[name]
      metrics.update(
        _scoped_metrics(
          "svd/ffn_down",
          scope=_layer_label(selected_metrics.layer, width=layer_label_width),
          metrics=selected_metrics,
        )
      )
    metrics.update(_summary_metrics("svd/ffn_down", layer_metrics.values()))
  return metrics


def _ffn_down_weight_names(names: list[str]) -> list[str]:
  return [
    name
    for name in names
    if name.startswith("blocks.") and name.endswith(".ffn.down.weight")
  ]


def _selected_layer_names(names: list[str]) -> tuple[str, ...]:
  last_index = len(names) - 1
  return tuple(
    dict.fromkeys((names[0], names[round(last_index * 0.25)], names[last_index]))
  )


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


def _svd_metrics(
  *,
  layer: int,
  singular_values: torch.Tensor,
) -> _LayerSvdMetrics:
  return _LayerSvdMetrics(
    layer=layer,
    effective_rank=_effective_rank(singular_values),
    stable_rank=_stable_rank(singular_values),
    r90=_energy_rank(singular_values, 0.90),
    r95=_energy_rank(singular_values, 0.95),
    r99=_energy_rank(singular_values, 0.99),
    energy_sum=float(singular_values.square().sum().item()),
  )


def _layer_label(layer: int, *, width: int) -> str:
  return f"layer_{layer:0{width}d}"


def _scoped_metrics(
  prefix: str,
  *,
  scope: str,
  metrics: _LayerSvdMetrics,
) -> dict[str, Scalar]:
  return {
    f"{prefix}/effective_rank/{scope}": metrics.effective_rank,
    f"{prefix}/stable_rank/{scope}": metrics.stable_rank,
    f"{prefix}/r90/{scope}": metrics.r90,
    f"{prefix}/r95/{scope}": metrics.r95,
    f"{prefix}/r99/{scope}": metrics.r99,
    f"{prefix}/energy_sum/{scope}": metrics.energy_sum,
  }


def _summary_metrics(
  prefix: str,
  layer_metrics: Iterable[_LayerSvdMetrics],
) -> dict[str, Scalar]:
  values = tuple(layer_metrics)
  metrics: dict[str, Scalar] = {}
  for name in ("effective_rank", "stable_rank", "r95", "r99"):
    series = [float(getattr(item, name)) for item in values]
    metrics.update(
      {
        f"{prefix}/{name}/min": min(series),
        f"{prefix}/{name}/max": max(series),
        f"{prefix}/{name}/mean": sum(series) / len(series),
      }
    )
  return metrics


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
