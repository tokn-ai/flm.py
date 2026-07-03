"""Training data builders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flm_datasets import (
  SourceCorpusConfig,
  TokenDataset,
  encode_text,
  iter_source_files,
  read_source_corpus,
)
from torch.utils.data import DataLoader

from flm_train.types import TrainConfig

_REPO_SOURCE_CACHE_VERSION = 1
_TOKEN_CACHE_DTYPE = "int32"


@dataclass(frozen=True)
class RepoSourceDatasetBundle:
  dataloader: DataLoader
  token_count: int
  file_count: int


@dataclass(frozen=True)
class RepoSourceCachePaths:
  tokens: Path
  metadata: Path


def build_repo_source_dataset(config: TrainConfig) -> RepoSourceDatasetBundle:
  corpus_config = SourceCorpusConfig(root=config.data.repo_root)
  source_files = iter_source_files(corpus_config)
  file_count = len(source_files)
  cache_paths = _repo_source_cache_paths(
    root=config.data.repo_root,
    cache_dir=config.data.cache_dir,
    encoding_name=config.data.encoding_name,
    source_files=source_files,
  )
  tokens = _read_token_cache(cache_paths)
  if tokens is None:
    corpus = read_source_corpus(corpus_config, paths=source_files)
    tokens = encode_text(corpus, encoding_name=config.data.encoding_name)
    _write_token_cache(
      cache_paths,
      tokens=tokens,
      encoding_name=config.data.encoding_name,
      file_count=file_count,
    )
  dataset = TokenDataset(tokens, seq_len=config.data.seq_len)
  dataloader = DataLoader(
    dataset,
    batch_size=config.loop.batch_size,
    shuffle=True,
    drop_last=False,
  )
  return RepoSourceDatasetBundle(
    dataloader=dataloader,
    token_count=len(tokens),
    file_count=file_count,
  )


def _repo_source_cache_paths(
  *,
  root: Path,
  cache_dir: Path | None,
  encoding_name: str,
  source_files: list[Path],
) -> RepoSourceCachePaths | None:
  if cache_dir is None:
    return None
  root = root.resolve()
  resolved_cache_dir = cache_dir if cache_dir.is_absolute() else root / cache_dir
  digest = _repo_source_cache_digest(
    root=root,
    encoding_name=encoding_name,
    source_files=source_files,
  )
  stem = f"repo_sources-{digest}"
  return RepoSourceCachePaths(
    tokens=resolved_cache_dir / f"{stem}.npy",
    metadata=resolved_cache_dir / f"{stem}.json",
  )


def _repo_source_cache_digest(
  *,
  root: Path,
  encoding_name: str,
  source_files: list[Path],
) -> str:
  manifest = {
    "version": _REPO_SOURCE_CACHE_VERSION,
    "encoding_name": encoding_name,
    "files": [
      {
        "path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
      }
      for path in source_files
      for stat in [path.stat()]
    ],
  }
  payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _read_token_cache(paths: RepoSourceCachePaths | None) -> list[int] | None:
  if paths is None or not paths.tokens.exists() or not paths.metadata.exists():
    return None
  try:
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return None
  if not _is_token_cache_metadata(metadata):
    return None
  if metadata["tokens_file"] != paths.tokens.name:
    return None
  try:
    token_array = np.load(paths.tokens, allow_pickle=False)
  except ValueError:
    return None
  if token_array.dtype != np.dtype(_TOKEN_CACHE_DTYPE):
    return None
  if token_array.ndim != 1:
    return None
  if int(metadata["token_count"]) != int(token_array.shape[0]):
    return None
  return token_array.astype(np.int64).tolist()


def _write_token_cache(
  paths: RepoSourceCachePaths | None,
  *,
  tokens: list[int],
  encoding_name: str,
  file_count: int,
) -> None:
  if paths is None:
    return
  paths.tokens.parent.mkdir(parents=True, exist_ok=True)
  token_array = np.asarray(tokens, dtype=np.int32)
  np.save(paths.tokens, token_array)
  paths.metadata.write_text(
    json.dumps(
      {
        "version": _REPO_SOURCE_CACHE_VERSION,
        "encoding_name": encoding_name,
        "dtype": _TOKEN_CACHE_DTYPE,
        "token_count": int(token_array.shape[0]),
        "file_count": file_count,
        "tokens_file": paths.tokens.name,
      },
      indent=2,
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )


def _is_token_cache_metadata(metadata: object) -> bool:
  return (
    isinstance(metadata, dict)
    and metadata.get("version") == _REPO_SOURCE_CACHE_VERSION
    and metadata.get("dtype") == _TOKEN_CACHE_DTYPE
    and isinstance(metadata.get("token_count"), int)
    and isinstance(metadata.get("file_count"), int)
    and isinstance(metadata.get("tokens_file"), str)
  )
