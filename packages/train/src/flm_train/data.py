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
  tokens_path: Path
  token_count: int
  file_count: int


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
  return RepoSourceDatasetBundle(
    dataloader=dataloader,
    token_count=int(metadata["token_count"]),
    file_count=int(metadata["file_count"]),
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
    resolved_version=version,
  )


def publish_repo_source_dataset(
  *,
  repo_root: Path,
  dataset_root: Path,
  encoding_name: str = "cl100k_base",
) -> PublishedDatasetInfo:
  corpus_config = SourceCorpusConfig(root=repo_root)
  source_files = iter_source_files(corpus_config)
  root = repo_root.resolve()
  file_records = _source_file_records(root=root, source_files=source_files)
  version = _published_dataset_digest(
    encoding_name=encoding_name,
    file_records=file_records,
  )
  version_dir = dataset_root / "versions" / version
  tokens_path = version_dir / "tokens.npy"
  manifest_path = version_dir / "manifest.json"
  files_path = version_dir / "files.jsonl"

  if not manifest_path.exists() or not tokens_path.exists() or not files_path.exists():
    corpus = read_source_corpus(corpus_config, paths=source_files)
    tokens = encode_text(corpus, encoding_name=encoding_name)
    version_dir.mkdir(parents=True, exist_ok=True)
    np.save(tokens_path, np.asarray(tokens, dtype=np.int32))
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
        "token_count": len(tokens),
        "file_count": len(source_files),
        "tokens_file": tokens_path.name,
        "files_file": files_path.name,
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
    tokens_path=tokens_path,
    token_count=int(manifest["token_count"]),
    file_count=int(manifest["file_count"]),
  )


def _source_file_records(
  *,
  root: Path,
  source_files: list[Path],
) -> list[dict[str, int | str]]:
  return [
    {
      "path": path.relative_to(root).as_posix(),
      "size": stat.st_size,
      "mtime_ns": stat.st_mtime_ns,
    }
    for path in source_files
    for stat in [path.stat()]
  ]


def _published_dataset_digest(
  *,
  encoding_name: str,
  file_records: list[dict[str, int | str]],
) -> str:
  manifest = {
    "format_version": _PUBLISHED_DATASET_VERSION,
    "encoding_name": encoding_name,
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
  if not isinstance(manifest.get("tokens_file"), str):
    raise ValueError(f"invalid dataset manifest: {manifest_path}")
  return manifest


def _load_dataset_tokens(config: DataConfig, manifest: dict[str, Any]) -> np.ndarray:
  resolved_version = config.resolved_version or _resolve_dataset_version(
    config.dataset_root,
    config.version,
  )
  tokens_path = (
    config.dataset_root / "versions" / resolved_version / str(manifest["tokens_file"])
  )
  token_array = np.load(tokens_path, allow_pickle=False)
  if token_array.dtype != np.dtype(_TOKEN_CACHE_DTYPE):
    raise ValueError(f"unsupported token dtype: {tokens_path}")
  if token_array.ndim != 1:
    raise ValueError(f"token dataset must be a 1D array: {tokens_path}")
  if int(manifest["token_count"]) != int(token_array.shape[0]):
    raise ValueError(f"token count mismatch: {tokens_path}")
  return token_array


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
