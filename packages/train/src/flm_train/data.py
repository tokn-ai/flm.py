"""Training data builders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from flm_datasets import (
  SourceCorpusConfig,
  TokenDataset,
  encode_text,
  iter_source_files,
  read_source_corpus,
)
from torch.utils.data import DataLoader

from flm_train.types import DataConfig, TrainConfig

_PUBLISHED_DATASET_VERSION = 1
_TOKEN_CACHE_DTYPE = "int32"


@dataclass(frozen=True)
class RepoSourceDatasetBundle:
  dataloader: DataLoader
  token_count: int
  file_count: int


@dataclass(frozen=True)
class PublishedDatasetInfo:
  dataset_root: Path
  version: str
  manifest_path: Path
  split_paths: dict[str, Path]
  token_count: int
  file_count: int
  splits: dict[str, dict[str, int]]


def build_training_dataset(config: TrainConfig) -> RepoSourceDatasetBundle:
  if config.data.kind == "token_dataset":
    return build_token_dataset(config)
  raise ValueError(f"unsupported data.kind: {config.data.kind}")


def build_token_dataset(config: TrainConfig) -> RepoSourceDatasetBundle:
  resolved_data = resolve_data_config(config.data)
  metadata = _load_dataset_manifest(resolved_data)
  token_array = _load_dataset_tokens(resolved_data, metadata)
  tokens = token_array.astype(np.int64).tolist()
  dataset = TokenDataset(tokens, seq_len=resolved_data.seq_len)
  dataloader = DataLoader(
    dataset,
    batch_size=config.loop.batch_size,
    shuffle=True,
    drop_last=False,
  )
  split_metadata = _split_metadata(metadata, resolved_data.split)
  return RepoSourceDatasetBundle(
    dataloader=dataloader,
    token_count=int(split_metadata["token_count"]),
    file_count=int(split_metadata["file_count"]),
  )


def resolve_data_config(config: DataConfig) -> DataConfig:
  if config.kind != "token_dataset":
    return config
  version = _resolve_dataset_version(config.dataset_root, config.version)
  return DataConfig(
    kind=config.kind,
    encoding_name=config.encoding_name,
    seq_len=config.seq_len,
    dataset_root=config.dataset_root,
    version=config.version,
    split=config.split,
    resolved_version=version,
  )


def publish_repo_source_dataset(
  *,
  repo_root: Path,
  dataset_root: Path,
  encoding_name: str = "cl100k_base",
  train_ratio: float = 0.98,
  val_ratio: float = 0.01,
  test_ratio: float = 0.01,
  split_seed: int = 42,
) -> PublishedDatasetInfo:
  _validate_split_ratios(
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
  )
  corpus_config = SourceCorpusConfig(root=repo_root)
  source_files = iter_source_files(corpus_config)
  root = repo_root.resolve()
  file_records = _source_file_records(
    root=root,
    source_files=source_files,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    split_seed=split_seed,
  )
  version = _published_dataset_digest(
    encoding_name=encoding_name,
    split_seed=split_seed,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
    file_records=file_records,
  )
  version_dir = dataset_root / "versions" / version
  manifest_path = version_dir / "manifest.json"
  files_path = version_dir / "files.jsonl"
  split_paths = {
    "train": version_dir / "train.npy",
    "val": version_dir / "val.npy",
    "test": version_dir / "test.npy",
  }

  if (
    not manifest_path.exists()
    or not files_path.exists()
    or any(not path.exists() for path in split_paths.values())
  ):
    split_source_files = _split_source_files(source_files, file_records)
    split_metadata = {}
    version_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_files in split_source_files.items():
      corpus = read_source_corpus(corpus_config, paths=split_files)
      tokens = encode_text(corpus, encoding_name=encoding_name)
      np.save(split_paths[split_name], np.asarray(tokens, dtype=np.int32))
      split_metadata[split_name] = {
        "tokens_file": split_paths[split_name].name,
        "token_count": len(tokens),
        "file_count": len(split_files),
      }
    files_path.write_text(
      "\n".join(json.dumps(record, sort_keys=True) for record in file_records) + "\n",
      encoding="utf-8",
    )
    _write_json(
      manifest_path,
      {
        "version": version,
        "format_version": _PUBLISHED_DATASET_VERSION,
        "kind": "repo_sources",
        "repo_root": str(root),
        "encoding_name": encoding_name,
        "dtype": _TOKEN_CACHE_DTYPE,
        "token_count": sum(
          int(metadata["token_count"]) for metadata in split_metadata.values()
        ),
        "file_count": len(source_files),
        "files_file": files_path.name,
        "split": {
          "strategy": "file_hash",
          "seed": split_seed,
          "train": train_ratio,
          "val": val_ratio,
          "test": test_ratio,
        },
        "splits": split_metadata,
        "created_at": datetime.now(UTC).isoformat(),
      },
    )
  manifest = _read_json(manifest_path)
  _write_json(
    dataset_root / "latest.json",
    {
      "version": version,
      "manifest": f"versions/{version}/manifest.json",
      "updated_at": datetime.now(UTC).isoformat(),
    },
  )
  return PublishedDatasetInfo(
    dataset_root=dataset_root,
    version=version,
    manifest_path=manifest_path,
    split_paths=split_paths,
    token_count=int(manifest["token_count"]),
    file_count=int(manifest["file_count"]),
    splits={
      name: {
        "token_count": int(metadata["token_count"]),
        "file_count": int(metadata["file_count"]),
      }
      for name, metadata in manifest["splits"].items()
    },
  )


def _source_file_records(
  *,
  root: Path,
  source_files: list[Path],
  train_ratio: float,
  val_ratio: float,
  split_seed: int,
) -> list[dict[str, int | str]]:
  return [
    {
      "path": path.relative_to(root).as_posix(),
      "size": stat.st_size,
      "mtime_ns": stat.st_mtime_ns,
      "split": _assign_file_split(
        path.relative_to(root).as_posix(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
      ),
    }
    for path in source_files
    for stat in [path.stat()]
  ]


def _published_dataset_digest(
  *,
  encoding_name: str,
  split_seed: int,
  train_ratio: float,
  val_ratio: float,
  test_ratio: float,
  file_records: list[dict[str, int | str]],
) -> str:
  manifest = {
    "format_version": _PUBLISHED_DATASET_VERSION,
    "encoding_name": encoding_name,
    "split": {
      "strategy": "file_hash",
      "seed": split_seed,
      "train": train_ratio,
      "val": val_ratio,
      "test": test_ratio,
    },
    "files": file_records,
  }
  payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_dataset_version(dataset_root: Path, version: str) -> str:
  if version != "latest":
    return version
  latest_path = dataset_root / "latest.json"
  latest = _read_json(latest_path)
  resolved_version = latest.get("version")
  if not isinstance(resolved_version, str) or not resolved_version:
    raise ValueError(f"invalid dataset latest pointer: {latest_path}")
  return resolved_version


def _load_dataset_manifest(config: DataConfig) -> dict[str, Any]:
  resolved_version = config.resolved_version or _resolve_dataset_version(
    config.dataset_root,
    config.version,
  )
  manifest_path = config.dataset_root / "versions" / resolved_version / "manifest.json"
  manifest = _read_json(manifest_path)
  if manifest.get("format_version") != _PUBLISHED_DATASET_VERSION:
    raise ValueError(f"unsupported dataset format: {manifest_path}")
  if manifest.get("dtype") != _TOKEN_CACHE_DTYPE:
    raise ValueError(f"unsupported dataset dtype: {manifest_path}")
  if manifest.get("encoding_name") != config.encoding_name:
    raise ValueError(
      f"dataset encoding {manifest.get('encoding_name')} does not match "
      f"config encoding {config.encoding_name}"
    )
  split_metadata = _split_metadata(manifest, config.split)
  if not isinstance(split_metadata.get("tokens_file"), str):
    raise ValueError(f"invalid dataset manifest: {manifest_path}")
  return manifest


def _load_dataset_tokens(config: DataConfig, manifest: dict[str, Any]) -> np.ndarray:
  resolved_version = config.resolved_version or _resolve_dataset_version(
    config.dataset_root,
    config.version,
  )
  split_metadata = _split_metadata(manifest, config.split)
  tokens_path = (
    config.dataset_root
    / "versions"
    / resolved_version
    / str(split_metadata["tokens_file"])
  )
  token_array = np.load(tokens_path, allow_pickle=False)
  if token_array.dtype != np.dtype(_TOKEN_CACHE_DTYPE):
    raise ValueError(f"unsupported token dtype: {tokens_path}")
  if token_array.ndim != 1:
    raise ValueError(f"token dataset must be a 1D array: {tokens_path}")
  if int(split_metadata["token_count"]) != int(token_array.shape[0]):
    raise ValueError(f"token count mismatch: {tokens_path}")
  return token_array


def _split_metadata(manifest: dict[str, Any], split: str) -> dict[str, Any]:
  splits = manifest.get("splits")
  if not isinstance(splits, dict):
    raise ValueError("dataset manifest does not define splits")
  metadata = splits.get(split)
  if not isinstance(metadata, dict):
    raise ValueError(f"dataset split not found: {split}")
  return metadata


def _validate_split_ratios(
  *,
  train_ratio: float,
  val_ratio: float,
  test_ratio: float,
) -> None:
  if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
    raise ValueError("split ratios must be non-negative")
  total = train_ratio + val_ratio + test_ratio
  if abs(total - 1.0) > 1e-9:
    raise ValueError("split ratios must sum to 1.0")


def _assign_file_split(
  relative_path: str,
  *,
  train_ratio: float,
  val_ratio: float,
  split_seed: int,
) -> str:
  payload = f"{split_seed}:{relative_path}".encode()
  value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
  if value < train_ratio:
    return "train"
  if value < train_ratio + val_ratio:
    return "val"
  return "test"


def _split_source_files(
  source_files: list[Path],
  file_records: list[dict[str, int | str]],
) -> dict[str, list[Path]]:
  split_files: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
  for path, record in zip(source_files, file_records, strict=True):
    split_files[str(record["split"])].append(path)
  return split_files


def _read_json(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
