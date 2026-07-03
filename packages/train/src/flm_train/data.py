"""Training data builders."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class RepoSourceDatasetBundle:
  dataloader: DataLoader
  token_count: int
  file_count: int


def build_repo_source_dataset(config: TrainConfig) -> RepoSourceDatasetBundle:
  corpus_config = SourceCorpusConfig(root=config.data.repo_root)
  source_files = iter_source_files(corpus_config)
  file_count = len(source_files)
  cache_path = _repo_source_cache_path(
    root=config.data.repo_root,
    cache_dir=config.data.cache_dir,
    encoding_name=config.data.encoding_name,
    source_files=source_files,
  )
  tokens = _read_token_cache(cache_path)
  if tokens is None:
    corpus = read_source_corpus(corpus_config, paths=source_files)
    tokens = encode_text(corpus, encoding_name=config.data.encoding_name)
    _write_token_cache(cache_path, tokens)
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


def _repo_source_cache_path(
  *,
  root: Path,
  cache_dir: Path | None,
  encoding_name: str,
  source_files: list[Path],
) -> Path | None:
  if cache_dir is None:
    return None
  root = root.resolve()
  resolved_cache_dir = cache_dir if cache_dir.is_absolute() else root / cache_dir
  digest = _repo_source_cache_digest(
    root=root,
    encoding_name=encoding_name,
    source_files=source_files,
  )
  return resolved_cache_dir / f"repo_sources-{digest}.pkl"


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


def _read_token_cache(path: Path | None) -> list[int] | None:
  if path is None or not path.exists():
    return None
  with path.open("rb") as file:
    payload = pickle.load(file)
  if not _is_token_cache_payload(payload):
    return None
  if payload["version"] != _REPO_SOURCE_CACHE_VERSION:
    return None
  return payload["tokens"]


def _write_token_cache(path: Path | None, tokens: list[int]) -> None:
  if path is None:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as file:
    pickle.dump(
      {
        "version": _REPO_SOURCE_CACHE_VERSION,
        "tokens": tokens,
      },
      file,
      protocol=pickle.HIGHEST_PROTOCOL,
    )


def _is_token_cache_payload(payload: Any) -> bool:
  return (
    isinstance(payload, dict)
    and isinstance(payload.get("version"), int)
    and isinstance(payload.get("tokens"), list)
    and all(isinstance(token, int) for token in payload["tokens"])
  )
