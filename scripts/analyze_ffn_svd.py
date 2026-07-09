"""Analyze SVD spectra and ranks of FFN weights in FLM checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MatrixReport:
  name: str
  shape: tuple[int, int]
  rank: int
  effective_rank: float
  stable_rank: float
  r90: int
  r95: int
  r99: int
  top_singular_values: tuple[float, ...]


def main() -> None:
  args = _parse_args()
  checkpoint = _resolve_checkpoint_path(args.checkpoint)
  metadata = json.loads((checkpoint / "model_state.json").read_text(encoding="utf-8"))
  arrays = np.load(checkpoint / "model.npz")

  reports: list[MatrixReport] = []
  for name in sorted(_ffn_weight_names(arrays.files), key=_sort_key):
    if args.layer is not None and _layer_index(name) not in set(args.layer):
      continue
    tensor = _load_tensor(name, arrays, metadata)
    reports.extend(
      _matrix_views(
        name,
        tensor,
        rank_rtol=args.rank_rtol,
        top_k=args.top_k,
      )
    )

  print(f"checkpoint={checkpoint}")
  print("name,shape,rank,effective_rank,stable_rank,r90,r95,r99,top_singular_values")
  for report in reports:
    print(
      f"{report.name},"
      f"{report.shape[0]}x{report.shape[1]},"
      f"{report.rank},"
      f"{report.effective_rank:.2f},"
      f"{report.stable_rank:.2f},"
      f"{report.r90},"
      f"{report.r95},"
      f"{report.r99},"
      f"{_format_values(report.top_singular_values)}"
    )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path(
      "runs/100mib_4k_repo_reference/20260704-160611-092c2a/checkpoints/latest"
    ),
  )
  parser.add_argument("--layer", type=int, action="append")
  parser.add_argument("--top-k", type=int, default=5)
  parser.add_argument("--rank-rtol", type=float, default=1e-5)
  return parser.parse_args()


def _resolve_checkpoint_path(path: Path) -> Path:
  if path.name == "latest":
    return path.parent / path.read_text(encoding="utf-8").strip()
  return path


def _ffn_weight_names(names: list[str]) -> list[str]:
  return [
    name
    for name in names
    if name.startswith("blocks.")
    and ".ffn." in name
    and name.endswith(".weight")
    and (".up." in name or ".down." in name)
  ]


def _sort_key(name: str) -> tuple[int, int]:
  projection_order = 0 if ".up." in name else 1
  return (_layer_index(name), projection_order)


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


def _matrix_views(
  name: str,
  tensor: torch.Tensor,
  *,
  rank_rtol: float,
  top_k: int,
) -> list[MatrixReport]:
  if ".up." not in name:
    return [_analyze_matrix(name, tensor, rank_rtol=rank_rtol, top_k=top_k)]

  gate, value = tensor.chunk(2, dim=0)
  return [
    _analyze_matrix(name, tensor, rank_rtol=rank_rtol, top_k=top_k),
    _analyze_matrix(
      name.replace(".up.", ".gate."),
      gate,
      rank_rtol=rank_rtol,
      top_k=top_k,
    ),
    _analyze_matrix(
      name.replace(".up.", ".value."),
      value,
      rank_rtol=rank_rtol,
      top_k=top_k,
    ),
  ]


def _analyze_matrix(
  name: str,
  matrix: torch.Tensor,
  *,
  rank_rtol: float,
  top_k: int,
) -> MatrixReport:
  singular_values = torch.linalg.svdvals(matrix).cpu()
  max_singular = singular_values.max().item()
  threshold = matrix.shape[1] * max_singular * rank_rtol
  rank = int((singular_values > threshold).sum().item())
  squared = singular_values.square()
  energy = squared / squared.sum()
  cumulative = torch.cumsum(energy, dim=0)
  entropy = -(
    energy * torch.log(energy.clamp_min(torch.finfo(energy.dtype).tiny))
  ).sum()
  stable_rank = squared.sum().item() / (max_singular * max_singular)
  return MatrixReport(
    name=name,
    shape=tuple(matrix.shape),
    rank=rank,
    effective_rank=float(torch.exp(entropy).item()),
    stable_rank=stable_rank,
    r90=_energy_rank(cumulative, 0.90),
    r95=_energy_rank(cumulative, 0.95),
    r99=_energy_rank(cumulative, 0.99),
    top_singular_values=tuple(
      float(value) for value in singular_values[:top_k].tolist()
    ),
  )


def _energy_rank(cumulative: torch.Tensor, target: float) -> int:
  return int(torch.searchsorted(cumulative, torch.tensor(target)).item()) + 1


def _format_values(values: tuple[float, ...]) -> str:
  return "[" + " ".join(f"{value:.4g}" for value in values) + "]"


if __name__ == "__main__":
  main()
