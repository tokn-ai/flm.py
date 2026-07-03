"""Safe checkpoint serialization for training state."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointState:
  step: int
  tokens_seen: int


def save_checkpoint(
  *,
  checkpoint_dir: Path,
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  state: CheckpointState,
) -> Path:
  path = checkpoint_dir / f"step-{state.step:08d}"
  path.mkdir(parents=True, exist_ok=True)

  model_payload = _encode_state(model.state_dict())
  optimizer_payload = _encode_state(optimizer.state_dict())
  _write_npz(path / "model.npz", model_payload.arrays)
  _write_npz(path / "optimizer.npz", optimizer_payload.arrays)
  _write_json(path / "model_state.json", model_payload.metadata)
  _write_json(path / "optimizer_state.json", optimizer_payload.metadata)
  _write_json(path / "trainer_state.json", asdict(state))
  _write_json(
    path / "manifest.json",
    {
      "format": "flm-checkpoint-v1",
      "step": state.step,
      "tokens_seen": state.tokens_seen,
      "model": "model.npz",
      "optimizer": "optimizer.npz",
      "trainer_state": "trainer_state.json",
    },
  )
  _write_latest_marker(checkpoint_dir, path)
  return path


def load_checkpoint(
  *,
  path: Path,
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  map_location: str,
) -> CheckpointState:
  checkpoint_path = resolve_checkpoint_path(path)
  trainer_state = json.loads(
    (checkpoint_path / "trainer_state.json").read_text(encoding="utf-8")
  )
  model_state = _decode_state(
    metadata=json.loads(
      (checkpoint_path / "model_state.json").read_text(encoding="utf-8")
    ),
    arrays=_read_npz(checkpoint_path / "model.npz"),
    map_location=map_location,
  )
  optimizer_state = _decode_state(
    metadata=json.loads(
      (checkpoint_path / "optimizer_state.json").read_text(encoding="utf-8")
    ),
    arrays=_read_npz(checkpoint_path / "optimizer.npz"),
    map_location=map_location,
  )
  model.load_state_dict(model_state)
  optimizer.load_state_dict(optimizer_state)
  return CheckpointState(
    step=int(trainer_state["step"]),
    tokens_seen=int(trainer_state["tokens_seen"]),
  )


def resolve_checkpoint_path(path: Path) -> Path:
  if path.is_dir():
    return path
  if path.name == "latest":
    target = path.read_text(encoding="utf-8").strip()
    return path.parent / target
  raise FileNotFoundError(path)


def latest_checkpoint_path(checkpoint_dir: Path) -> Path | None:
  latest = checkpoint_dir / "latest"
  if not latest.is_file():
    return None
  path = resolve_checkpoint_path(latest)
  if not path.is_dir():
    return None
  return path


def prune_checkpoints(*, checkpoint_dir: Path, keep_last: int) -> None:
  if keep_last <= 0:
    for path in _checkpoint_paths(checkpoint_dir):
      shutil.rmtree(path)
    return
  paths = _checkpoint_paths(checkpoint_dir)
  for path in paths[:-keep_last]:
    shutil.rmtree(path)


@dataclass(frozen=True)
class _EncodedState:
  metadata: Any
  arrays: dict[str, np.ndarray]


def _encode_state(value: Any) -> _EncodedState:
  arrays: dict[str, np.ndarray] = {}

  def encode(item: Any) -> Any:
    if isinstance(item, torch.Tensor):
      name = f"tensor_{len(arrays)}"
      arrays[name] = item.detach().cpu().numpy()
      return {"__tensor__": name}
    if isinstance(item, dict):
      return {str(key): encode(value) for key, value in item.items()}
    if isinstance(item, list):
      return [encode(value) for value in item]
    if isinstance(item, tuple):
      return {"__tuple__": [encode(value) for value in item]}
    return item

  return _EncodedState(metadata=encode(value), arrays=arrays)


def _decode_state(
  *,
  metadata: Any,
  arrays,
  map_location: str,
) -> Any:
  device = torch.device(map_location)

  def decode(item: Any) -> Any:
    if isinstance(item, dict):
      if "__tensor__" in item:
        return torch.from_numpy(arrays[item["__tensor__"]]).to(device)
      if "__tuple__" in item:
        return tuple(decode(value) for value in item["__tuple__"])
      return {_restore_key(key): decode(value) for key, value in item.items()}
    if isinstance(item, list):
      return [decode(value) for value in item]
    return item

  return decode(metadata)


def _restore_key(key: str) -> str | int:
  if key.isdecimal():
    return int(key)
  return key


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
  np.savez(path, **arrays)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as arrays:
    return {key: arrays[key] for key in arrays.files}


def _write_json(path: Path, payload: object) -> None:
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _write_latest_marker(checkpoint_dir: Path, path: Path) -> None:
  (checkpoint_dir / "latest").write_text(path.name + "\n", encoding="utf-8")


def _checkpoint_paths(checkpoint_dir: Path) -> list[Path]:
  if not checkpoint_dir.is_dir():
    return []
  return sorted(
    path
    for path in checkpoint_dir.iterdir()
    if path.is_dir() and path.name.startswith("step-")
  )
