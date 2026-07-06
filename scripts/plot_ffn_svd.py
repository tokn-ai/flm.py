"""Plot FFN SVD diagnostics for FLM checkpoints."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


MATRIX_VIEWS = ("gate", "up", "down")


@dataclass(frozen=True)
class MatrixSpectrum:
  layer: int
  view: str
  singular_values: torch.Tensor

  @property
  def full_rank(self) -> int:
    return int(self.singular_values.numel())

  @property
  def energy_sum(self) -> float:
    return float(self.singular_values.square().sum().item())

  @property
  def stable_rank(self) -> float:
    squared = self.singular_values.square()
    return float((squared.sum() / squared.max()).item())

  @property
  def effective_rank(self) -> float:
    squared = self.singular_values.square()
    energy = squared / squared.sum()
    entropy = -(
      energy * torch.log(energy.clamp_min(torch.finfo(energy.dtype).tiny))
    ).sum()
    return float(torch.exp(entropy).item())

  def energy_rank(self, target: float) -> int:
    squared = self.singular_values.square()
    cumulative = torch.cumsum(squared / squared.sum(), dim=0)
    return int(torch.searchsorted(cumulative, torch.tensor(target)).item()) + 1


def main() -> None:
  args = _parse_args()
  checkpoint = _resolve_checkpoint_path(args.checkpoint)
  spectra = _load_ffn_spectra(checkpoint)
  args.outdir.mkdir(parents=True, exist_ok=True)

  for view in MATRIX_VIEWS:
    _plot_rank_metrics(
      [spectrum for spectrum in spectra if spectrum.view == view],
      outpath=args.outdir / f"rank_{view}.png",
    )
  _plot_energy(spectra, args.outdir / "energy.png")
  for view in MATRIX_VIEWS:
    _plot_heatmap(
      [spectrum for spectrum in spectra if spectrum.view == view],
      outpath=args.outdir / f"svd_heatmap_{view}.png",
      heatmap_vmin=args.heatmap_vmin,
    )
  print(f"checkpoint={checkpoint}")
  print(f"outdir={args.outdir}")
  for filename in [
    "rank_gate.png",
    "rank_up.png",
    "rank_down.png",
    "energy.png",
    "svd_heatmap_gate.png",
    "svd_heatmap_up.png",
    "svd_heatmap_down.png",
  ]:
    print(args.outdir / filename)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path(
      "runs/100mib_4k_repo_reference/20260704-160611-092c2a/checkpoints/latest"
    ),
  )
  parser.add_argument(
    "--outdir",
    type=Path,
    default=Path(
      "runs/100mib_4k_repo_reference/20260704-160611-092c2a/svd"
    ),
  )
  parser.add_argument(
    "--heatmap-vmin",
    type=float,
    help=(
      "minimum log10(s_i / s_0) color value. Defaults to each view's "
      "1st percentile."
    ),
  )
  return parser.parse_args()


def _resolve_checkpoint_path(path: Path) -> Path:
  if path.name == "latest":
    return path.parent / path.read_text(encoding="utf-8").strip()
  return path


def _load_ffn_spectra(checkpoint: Path) -> list[MatrixSpectrum]:
  metadata = json.loads((checkpoint / "model_state.json").read_text(encoding="utf-8"))
  arrays = np.load(checkpoint / "model.npz")
  spectra: list[MatrixSpectrum] = []
  up_names = sorted(
    [name for name in arrays.files if name.endswith(".ffn.up.weight")],
    key=_layer_index,
  )
  for up_name in up_names:
    layer = _layer_index(up_name)
    up_weight = _load_tensor(up_name, arrays, metadata)
    gate_weight, value_weight = up_weight.chunk(2, dim=0)
    down_name = up_name.replace(".up.", ".down.")
    down_weight = _load_tensor(down_name, arrays, metadata)
    spectra.append(_spectrum(layer, "gate", gate_weight))
    spectra.append(_spectrum(layer, "up", value_weight))
    spectra.append(_spectrum(layer, "down", down_weight))
  return spectra


def _layer_index(name: str) -> int:
  return int(name.split(".")[1])


def _load_tensor(
  name: str,
  arrays: np.lib.npyio.NpzFile,
  metadata: dict[str, object],
) -> torch.Tensor:
  array = arrays[name]
  tensor_metadata = metadata[name]["__tensor__"]
  tensor = torch.from_numpy(array)
  if tensor_metadata["dtype"] == "torch.bfloat16":
    tensor = tensor.view(torch.bfloat16)
  return tensor.float()


def _spectrum(layer: int, view: str, matrix: torch.Tensor) -> MatrixSpectrum:
  return MatrixSpectrum(
    layer=layer,
    view=view,
    singular_values=torch.linalg.svdvals(matrix).cpu(),
  )


def _plot_rank_metrics(spectra: list[MatrixSpectrum], outpath: Path) -> None:
  spectra = sorted(spectra, key=lambda spectrum: spectrum.layer)
  layers = [spectrum.layer for spectrum in spectra]
  fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
  ax.plot(
    layers,
    [spectrum.effective_rank for spectrum in spectra],
    marker="o",
    label="effective",
  )
  ax.plot(
    layers,
    [spectrum.stable_rank for spectrum in spectra],
    marker="o",
    label="stable",
  )
  for target, label in [(0.90, "r90"), (0.95, "r95"), (0.99, "r99")]:
    ax.plot(
      layers,
      [spectrum.energy_rank(target) for spectrum in spectra],
      marker=".",
      label=label,
    )
  ax.axhline(spectra[0].full_rank, color="0.6", linestyle="--", label="full rank")
  ax.set_title(f"FFN {spectra[0].view} SVD rank metrics")
  ax.set_xlabel("layer")
  ax.set_ylabel("rank")
  ax.set_xticks(layers)
  ax.grid(alpha=0.25)
  ax.legend(loc="best")
  fig.tight_layout()
  fig.savefig(outpath)
  plt.close(fig)


def _plot_energy(spectra: list[MatrixSpectrum], outpath: Path) -> None:
  fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
  for view in MATRIX_VIEWS:
    view_spectra = sorted(
      [spectrum for spectrum in spectra if spectrum.view == view],
      key=lambda spectrum: spectrum.layer,
    )
    ax.plot(
      [spectrum.layer for spectrum in view_spectra],
      [spectrum.energy_sum for spectrum in view_spectra],
      marker="o",
      label=view,
    )
  ax.set_title("FFN spectral energy by layer")
  ax.set_xlabel("layer")
  ax.set_ylabel("sum of squared singular values")
  ax.grid(alpha=0.25)
  ax.legend(loc="best")
  fig.tight_layout()
  fig.savefig(outpath)
  plt.close(fig)


def _plot_heatmap(
  spectra: list[MatrixSpectrum],
  outpath: Path,
  heatmap_vmin: float | None,
) -> None:
  spectra = sorted(spectra, key=lambda spectrum: spectrum.layer)
  matrix = torch.stack(
    [
      torch.log10(
        spectrum.singular_values / spectrum.singular_values[0]
      ).clamp_min(-5.0)
      for spectrum in spectra
    ]
  ).T.numpy()
  vmin = float(np.quantile(matrix, 0.01)) if heatmap_vmin is None else heatmap_vmin
  fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
  image = ax.imshow(matrix, aspect="auto", origin="lower", vmin=vmin, vmax=0)
  ax.set_title(f"FFN {spectra[0].view} singular value spectrum")
  ax.set_xlabel("layer")
  ax.set_ylabel("singular index")
  ax.set_xticks(range(len(spectra)))
  ax.set_xticklabels([spectrum.layer for spectrum in spectra])
  fig.colorbar(image, ax=ax, label="log10(s_i / s_0)")
  fig.tight_layout()
  fig.savefig(outpath)
  plt.close(fig)


if __name__ == "__main__":
  main()
