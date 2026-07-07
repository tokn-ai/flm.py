"""Training data builders."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from flm_datasets import (
  SOURCE_CORPUS_SEPARATOR,
  ShardedTokenDataset,
  SourceCorpusConfig,
  TokenDataset,
  encode_text,
  get_tokenizer,
  iter_source_files,
  read_source_corpus,
  unitoken_encoding_name,
  unitoken_special_tokens,
  write_source_corpus_file,
)
from torch.utils.data import DataLoader

from flm_train.types import DataConfig, TrainConfig

_PUBLISHED_DATASET_VERSION = 1
_TOKEN_CACHE_DTYPE = "int32"
_FINEWEB_TOKEN_SHARD_SIZE = 256 * 1024 * 1024


@dataclass(frozen=True)
class RepoSourceDatasetBundle:
  dataloader: DataLoader
  token_count: int
  file_count: int
  byte_count: int


@dataclass(frozen=True)
class PublishedDatasetInfo:
  dataset_root: Path
  version: str
  manifest_path: Path
  split_paths: dict[str, Path | list[Path]]
  token_count: int
  file_count: int
  byte_count: int
  unigram_entropy_nats_per_token: float
  splits: dict[str, dict[str, int]]


def build_training_dataset(config: TrainConfig) -> RepoSourceDatasetBundle:
  if config.data.kind == "token_dataset":
    return build_token_dataset(config)
  raise ValueError(f"unsupported data.kind: {config.data.kind}")


def build_token_dataset(config: TrainConfig) -> RepoSourceDatasetBundle:
  resolved_data = resolve_data_config(config.data)
  metadata = _load_dataset_manifest(resolved_data)
  token_arrays = _load_dataset_token_arrays(resolved_data, metadata)
  dataset = (
    TokenDataset(token_arrays[0], seq_len=resolved_data.seq_len)
    if len(token_arrays) == 1
    else ShardedTokenDataset(token_arrays, seq_len=resolved_data.seq_len)
  )
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
    byte_count=int(split_metadata["byte_count"]),
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
  unitoken_vocab_size: int | None = None,
  unitoken_special_token_count: int = 16,
  tokenizer_root: Path = Path("tokenizers"),
  tokenizer_name: str | None = None,
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
  if unitoken_vocab_size is not None:
    if unitoken_special_token_count != 16:
      raise ValueError("unitoken currently supports exactly 16 reserved special tokens")
    name = tokenizer_name or f"repo_{unitoken_vocab_size}"
    tokenizer_path = tokenizer_root / name
    train_unitoken_tokenizer(
      source_files=source_files,
      corpus_config=corpus_config,
      tokenizer_path=tokenizer_path,
      vocab_size=unitoken_vocab_size,
      special_token_count=unitoken_special_token_count,
    )
    encoding_name = unitoken_encoding_name(tokenizer_path)
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
        "byte_count": sum(path.stat().st_size for path in split_files),
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
        "byte_count": sum(
          int(metadata["byte_count"]) for metadata in split_metadata.values()
        ),
        "unigram_entropy_nats_per_token": _token_entropy_nats_from_paths(
          split_paths.values()
        ),
        "files_file": files_path.name,
        "document_separator": SOURCE_CORPUS_SEPARATOR,
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
  if _manifest_missing_byte_counts(manifest):
    manifest = _manifest_with_byte_counts(manifest, file_records)
    _write_json(manifest_path, manifest)
  unigram_entropy_nats_per_token = float(
    manifest.get(
      "unigram_entropy_nats_per_token",
      _token_entropy_nats_from_paths(split_paths.values()),
    )
  )
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
    byte_count=int(manifest["byte_count"]),
    unigram_entropy_nats_per_token=unigram_entropy_nats_per_token,
    splits={
      name: {
        "token_count": int(metadata["token_count"]),
        "file_count": int(metadata["file_count"]),
        "byte_count": int(metadata["byte_count"]),
      }
      for name, metadata in manifest["splits"].items()
    },
  )


def publish_fineweb2_dataset(
  *,
  dataset_root: Path,
  config_name: str,
  encoding_name: str = "cl100k_base",
  dataset_name: str = "HuggingFaceFW/fineweb-2",
  source_split: str = "train",
  max_train_bytes: int = 50_000_000,
  max_val_bytes: int = 2_000_000,
  max_test_bytes: int = 2_000_000,
  train_ratio: float = 0.98,
  val_ratio: float = 0.01,
  test_ratio: float = 0.01,
  split_seed: int = 42,
  text_column: str = "text",
) -> PublishedDatasetInfo:
  _validate_split_ratios(
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
  )
  if max_train_bytes <= 0:
    raise ValueError("max_train_bytes must be positive")
  if max_val_bytes < 0 or max_test_bytes < 0:
    raise ValueError("max_val_bytes and max_test_bytes must be non-negative")

  split_byte_targets = {
    "train": max_train_bytes,
    "val": max_val_bytes,
    "test": max_test_bytes,
  }
  version = _published_fineweb2_digest(
    dataset_name=dataset_name,
    config_name=config_name,
    source_split=source_split,
    encoding_name=encoding_name,
    split_seed=split_seed,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
    split_byte_targets=split_byte_targets,
    text_column=text_column,
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
    split_tokens: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_metadata = {
      name: {
        "tokens_file": split_paths[name].name,
        "token_count": 0,
        "file_count": 0,
        "byte_count": 0,
      }
      for name in split_paths
    }
    records = []
    for index, row in enumerate(
      _iter_hf_dataset_rows(
        dataset_name=dataset_name,
        config_name=config_name,
        split=source_split,
      )
    ):
      text = str(row.get(text_column, ""))
      if not text:
        continue
      doc_id = str(row.get("id") or row.get("url") or index)
      split_name = _assign_file_split(
        doc_id,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
      )
      if split_metadata[split_name]["byte_count"] >= split_byte_targets[split_name]:
        if _fineweb2_byte_targets_reached(split_metadata, split_byte_targets):
          break
        continue
      byte_count = len(text.encode("utf-8"))
      tokens = encode_text(text + "\n\n", encoding_name=encoding_name)
      split_tokens[split_name].extend(tokens)
      split_metadata[split_name]["token_count"] += len(tokens)
      split_metadata[split_name]["file_count"] += 1
      split_metadata[split_name]["byte_count"] += byte_count
      records.append(
        {
          "index": index,
          "id": doc_id,
          "split": split_name,
          "byte_count": byte_count,
        }
      )
      if _fineweb2_byte_targets_reached(split_metadata, split_byte_targets):
        break

    version_dir.mkdir(parents=True, exist_ok=True)
    for split_name, tokens in split_tokens.items():
      np.save(split_paths[split_name], np.asarray(tokens, dtype=np.int32))
    files_path.write_text(
      "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
      encoding="utf-8",
    )
    _write_json(
      manifest_path,
      {
        "version": version,
        "format_version": _PUBLISHED_DATASET_VERSION,
        "kind": "fineweb2",
        "dataset_name": dataset_name,
        "config_name": config_name,
        "source_split": source_split,
        "encoding_name": encoding_name,
        "dtype": _TOKEN_CACHE_DTYPE,
        "token_count": sum(
          int(metadata["token_count"]) for metadata in split_metadata.values()
        ),
        "file_count": sum(
          int(metadata["file_count"]) for metadata in split_metadata.values()
        ),
        "byte_count": sum(
          int(metadata["byte_count"]) for metadata in split_metadata.values()
        ),
        "unigram_entropy_nats_per_token": _token_entropy_nats_from_paths(
          split_paths.values()
        ),
        "files_file": files_path.name,
        "split": {
          "strategy": "document_hash",
          "seed": split_seed,
          "train": train_ratio,
          "val": val_ratio,
          "test": test_ratio,
        },
        "limits": {
          "train_bytes": max_train_bytes,
          "val_bytes": max_val_bytes,
          "test_bytes": max_test_bytes,
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
    byte_count=int(manifest["byte_count"]),
    unigram_entropy_nats_per_token=float(manifest["unigram_entropy_nats_per_token"]),
    splits={
      name: {
        "token_count": int(metadata["token_count"]),
        "file_count": int(metadata["file_count"]),
        "byte_count": int(metadata["byte_count"]),
      }
      for name, metadata in manifest["splits"].items()
    },
  )


def publish_fineweb_parquet_dataset(
  *,
  source_root: Path,
  corpus_root: Path = Path("cache/corpora"),
  corpus_name: str = "fineweb_10bt",
  tokens_root: Path = Path("cache/tokens"),
  encoding_name: str = "cl100k_base",
  unitoken_vocab_size: int | None = None,
  unitoken_special_token_count: int = 16,
  tokenizer_root: Path = Path("tokenizers"),
  tokenizer_name: str | None = None,
  train_ratio: float = 0.8,
  val_ratio: float = 0.1,
  test_ratio: float = 0.1,
  split_seed: int = 42,
  text_column: str = "text",
  id_column: str = "id",
  parquet_batch_size: int = 1024,
) -> PublishedDatasetInfo:
  _validate_split_ratios(
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
  )
  parquet_files = sorted(source_root.rglob("*.parquet"))
  if not parquet_files:
    raise ValueError(f"no parquet files found under: {source_root}")
  tokenizer_name = tokenizer_name or (
    f"{corpus_name}_{unitoken_vocab_size}"
    if unitoken_vocab_size is not None
    else "cl100k_base"
  )
  if unitoken_vocab_size is not None:
    if unitoken_special_token_count != 16:
      raise ValueError("unitoken currently supports exactly 16 reserved special tokens")
    tokenizer_path = tokenizer_root / tokenizer_name
    train_unitoken_tokenizer_from_parquet(
      parquet_files=parquet_files,
      source_root=source_root,
      corpus_root=corpus_root,
      corpus_name=corpus_name,
      tokenizer_path=tokenizer_path,
      vocab_size=unitoken_vocab_size,
      special_token_count=unitoken_special_token_count,
      train_ratio=train_ratio,
      val_ratio=val_ratio,
      split_seed=split_seed,
      text_column=text_column,
      id_column=id_column,
      parquet_batch_size=parquet_batch_size,
    )
    encoding_name = unitoken_encoding_name(tokenizer_path)
  dataset_root = tokens_root / tokenizer_name / corpus_name
  tokenizer_fingerprint = _tokenizer_fingerprint(encoding_name)

  version = _published_fineweb_parquet_digest(
    source_root=source_root,
    parquet_files=parquet_files,
    encoding_name=encoding_name,
    tokenizer_fingerprint=tokenizer_fingerprint,
    split_seed=split_seed,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
    text_column=text_column,
    id_column=id_column,
  )
  manifest_path = dataset_root / "manifest.json"
  files_path = dataset_root / "files.jsonl"
  split_dirs = {
    "train": dataset_root / "train",
    "val": dataset_root / "val",
    "test": dataset_root / "test",
  }

  if (
    not manifest_path.exists()
    or not files_path.exists()
    or not _fineweb_parquet_shards_exist(manifest_path, dataset_root, version)
  ):
    dataset_root.mkdir(parents=True, exist_ok=True)
    _prepare_fineweb_split_dirs(split_dirs)
    written_metadata = _write_fineweb_parquet_token_shards(
      parquet_files=parquet_files,
      source_root=source_root,
      encoding_name=encoding_name,
      split_dirs=split_dirs,
      files_path=files_path,
      train_ratio=train_ratio,
      val_ratio=val_ratio,
      split_seed=split_seed,
      text_column=text_column,
      id_column=id_column,
      parquet_batch_size=parquet_batch_size,
      shard_size=_FINEWEB_TOKEN_SHARD_SIZE,
    )
    _write_json(
      manifest_path,
      {
        "version": version,
        "format_version": _PUBLISHED_DATASET_VERSION,
        "kind": "fineweb_parquet",
        "source_root": str(source_root),
        "corpus_name": corpus_name,
        "tokenizer_name": tokenizer_name,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "encoding_name": encoding_name,
        "dtype": _TOKEN_CACHE_DTYPE,
        "token_count": sum(
          int(metadata["token_count"]) for metadata in written_metadata.values()
        ),
        "file_count": sum(
          int(metadata["file_count"]) for metadata in written_metadata.values()
        ),
        "byte_count": sum(
          int(metadata["byte_count"]) for metadata in written_metadata.values()
        ),
        "unigram_entropy_nats_per_token": _token_entropy_nats_from_paths(
          _flatten_split_paths(dataset_root, written_metadata)
        ),
        "files_file": files_path.name,
        "document_separator": SOURCE_CORPUS_SEPARATOR,
        "token_shard_size": _FINEWEB_TOKEN_SHARD_SIZE,
        "split": {
          "strategy": "document_hash",
          "seed": split_seed,
          "train": train_ratio,
          "val": val_ratio,
          "test": test_ratio,
        },
        "columns": {
          "text": text_column,
          "id": id_column,
        },
        "splits": written_metadata,
        "created_at": datetime.now(UTC).isoformat(),
      },
    )

  manifest = _read_json(manifest_path)
  _write_json(
    dataset_root / "latest.json",
    {
      "version": version,
      "manifest": "manifest.json",
      "updated_at": datetime.now(UTC).isoformat(),
    },
  )
  return PublishedDatasetInfo(
    dataset_root=dataset_root,
    version=version,
    manifest_path=manifest_path,
    split_paths=_manifest_split_paths(dataset_root, manifest),
    token_count=int(manifest["token_count"]),
    file_count=int(manifest["file_count"]),
    byte_count=int(manifest["byte_count"]),
    unigram_entropy_nats_per_token=float(manifest["unigram_entropy_nats_per_token"]),
    splits={
      name: {
        "token_count": int(metadata["token_count"]),
        "file_count": int(metadata["file_count"]),
        "byte_count": int(metadata["byte_count"]),
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


def _published_fineweb2_digest(
  *,
  dataset_name: str,
  config_name: str,
  source_split: str,
  encoding_name: str,
  split_seed: int,
  train_ratio: float,
  val_ratio: float,
  test_ratio: float,
  split_byte_targets: dict[str, int],
  text_column: str,
) -> str:
  manifest = {
    "format_version": _PUBLISHED_DATASET_VERSION,
    "kind": "fineweb2",
    "dataset_name": dataset_name,
    "config_name": config_name,
    "source_split": source_split,
    "encoding_name": encoding_name,
    "split": {
      "strategy": "document_hash",
      "seed": split_seed,
      "train": train_ratio,
      "val": val_ratio,
      "test": test_ratio,
    },
    "limits": split_byte_targets,
    "text_column": text_column,
  }
  payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _iter_hf_dataset_rows(
  *,
  dataset_name: str,
  config_name: str,
  split: str,
):
  try:
    from datasets import load_dataset
  except ModuleNotFoundError as exc:
    raise ImportError("FineWeb2 publishing requires the datasets package") from exc
  return load_dataset(
    dataset_name,
    config_name,
    split=split,
    streaming=True,
  )


def _fineweb2_byte_targets_reached(
  split_metadata: dict[str, dict[str, int | str]],
  split_byte_targets: dict[str, int],
) -> bool:
  return all(
    int(split_metadata[split]["byte_count"]) >= target
    for split, target in split_byte_targets.items()
  )


def _token_entropy_nats_from_paths(paths) -> float:
  counts = np.zeros(0, dtype=np.int64)
  for path in paths:
    tokens = np.load(path, allow_pickle=False, mmap_mode="r")
    if tokens.size == 0:
      continue
    token_counts = np.bincount(tokens.astype(np.int64, copy=False))
    if token_counts.shape[0] > counts.shape[0]:
      counts = np.pad(counts, (0, token_counts.shape[0] - counts.shape[0]))
    counts[: token_counts.shape[0]] += token_counts
  total = int(counts.sum())
  if total == 0:
    return 0.0
  probabilities = counts[counts > 0].astype(np.float64) / total
  return float(-np.sum(probabilities * np.log(probabilities)))


def _manifest_missing_byte_counts(manifest: dict[str, Any]) -> bool:
  if "byte_count" not in manifest:
    return True
  splits = manifest.get("splits")
  if not isinstance(splits, dict):
    return True
  return any("byte_count" not in metadata for metadata in splits.values())


def _manifest_with_byte_counts(
  manifest: dict[str, Any],
  file_records: list[dict[str, int | str]],
) -> dict[str, Any]:
  byte_counts = {"train": 0, "val": 0, "test": 0}
  for record in file_records:
    split = str(record["split"])
    byte_counts[split] += int(record["size"])
  manifest = dict(manifest)
  manifest["byte_count"] = sum(byte_counts.values())
  manifest["splits"] = {
    name: {
      **metadata,
      "byte_count": byte_counts[name],
    }
    for name, metadata in manifest["splits"].items()
  }
  return manifest


def train_unitoken_tokenizer(
  *,
  source_files: list[Path],
  corpus_config: SourceCorpusConfig,
  tokenizer_path: Path,
  vocab_size: int,
  special_token_count: int = 16,
) -> None:
  if vocab_size <= 0:
    raise ValueError("unitoken vocab size must be positive")
  special_tokens = unitoken_special_tokens(special_token_count)
  if vocab_size <= 256 + len(special_tokens):
    raise ValueError(
      "unitoken vocab size must leave room for byte tokens and special tokens"
    )

  from uni_tokenizer import BpeTrainer, PreTokenizer

  tokenizer_path.mkdir(parents=True, exist_ok=True)
  corpus_path = tokenizer_path / "corpus.txt"
  write_source_corpus_file(corpus_path, corpus_config, paths=source_files)
  pre_tokenizer = PreTokenizer(special_tokens=special_tokens)
  words: dict[str, int] = {}
  for word, count in pre_tokenizer.get_words_from_file(corpus_path).items():
    words[word] = words.get(word, 0) + int(count)

  trainer = BpeTrainer(special_tokens)
  trainer.add_words(words)
  trainer.train(vocab_size=vocab_size)
  _save_unitoken_tokenizer(trainer, tokenizer_path)
  corpus_path.unlink(missing_ok=True)


def train_unitoken_tokenizer_from_parquet(
  *,
  parquet_files: list[Path],
  source_root: Path,
  corpus_root: Path,
  corpus_name: str,
  tokenizer_path: Path,
  vocab_size: int,
  special_token_count: int,
  train_ratio: float,
  val_ratio: float,
  split_seed: int,
  text_column: str,
  id_column: str,
  parquet_batch_size: int,
) -> None:
  if vocab_size <= 0:
    raise ValueError("unitoken vocab size must be positive")
  special_tokens = unitoken_special_tokens(special_token_count)
  if vocab_size <= 256 + len(special_tokens):
    raise ValueError(
      "unitoken vocab size must leave room for byte tokens and special tokens"
    )
  vocab_path = tokenizer_path / "vocab.json"
  merges_path = tokenizer_path / "merges.txt"
  if vocab_path.exists() and merges_path.exists():
    return

  from uni_tokenizer import BpeTrainer, PreTokenizer

  tokenizer_path.mkdir(parents=True, exist_ok=True)
  corpus_path = _build_fineweb_tokenizer_corpus(
    parquet_files=parquet_files,
    source_root=source_root,
    corpus_root=corpus_root,
    corpus_name=corpus_name,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    split_seed=split_seed,
    text_column=text_column,
    id_column=id_column,
    parquet_batch_size=parquet_batch_size,
  )
  corpus_manifest = _read_json(corpus_path.parent / "manifest.json")
  expected_manifest = _unitoken_tokenizer_manifest(
    tokenizer_name=tokenizer_path.name,
    vocab_size=vocab_size,
    special_token_count=special_token_count,
    corpus_fingerprint=str(corpus_manifest["fingerprint"]),
  )
  manifest_path = tokenizer_path / "manifest.json"
  if vocab_path.exists() and merges_path.exists() and manifest_path.exists():
    manifest = _read_json(manifest_path)
    if manifest.get("fingerprint") == expected_manifest["fingerprint"]:
      return

  pre_tokenizer = PreTokenizer(special_tokens=special_tokens)
  words = {
    word: int(count)
    for word, count in pre_tokenizer.get_words_from_file(corpus_path).items()
  }
  trainer = BpeTrainer(special_tokens)
  trainer.add_words(words)
  trainer.train(vocab_size=vocab_size)
  _save_unitoken_tokenizer(trainer, tokenizer_path, manifest=expected_manifest)


def _build_fineweb_tokenizer_corpus(
  *,
  parquet_files: list[Path],
  source_root: Path,
  corpus_root: Path,
  corpus_name: str,
  train_ratio: float,
  val_ratio: float,
  split_seed: int,
  text_column: str,
  id_column: str,
  parquet_batch_size: int,
) -> Path:
  corpus_dir = corpus_root / corpus_name
  corpus_path = corpus_dir / "corpus.txt"
  manifest_path = corpus_dir / "manifest.json"
  expected_manifest = _fineweb_tokenizer_corpus_manifest(
    source_root=source_root,
    parquet_files=parquet_files,
    corpus_name=corpus_name,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    split_seed=split_seed,
    text_column=text_column,
    id_column=id_column,
  )
  if corpus_path.exists() and manifest_path.exists():
    manifest = _read_json(manifest_path)
    if manifest.get("fingerprint") == expected_manifest["fingerprint"]:
      return corpus_path

  corpus_dir.mkdir(parents=True, exist_ok=True)
  tmp_path = corpus_dir / "corpus.txt.tmp"
  document_count = 0
  byte_count = 0
  with tmp_path.open("w", encoding="utf-8") as corpus_file:
    for record in _iter_fineweb_parquet_records(
      parquet_files=parquet_files,
      source_root=source_root,
      text_column=text_column,
      id_column=id_column,
      batch_size=parquet_batch_size,
    ):
      split_name = _assign_file_split(
        record["split_key"],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
      )
      if split_name != "train":
        continue
      corpus_file.write(record["text"])
      corpus_file.write(SOURCE_CORPUS_SEPARATOR)
      document_count += 1
      byte_count += int(record["byte_count"])
  os.replace(tmp_path, corpus_path)
  _write_json(
    manifest_path,
    {
      **expected_manifest,
      "document_count": document_count,
      "byte_count": byte_count,
      "created_at": datetime.now(UTC).isoformat(),
    },
  )
  return corpus_path


def _fineweb_tokenizer_corpus_manifest(
  *,
  source_root: Path,
  parquet_files: list[Path],
  corpus_name: str,
  train_ratio: float,
  val_ratio: float,
  split_seed: int,
  text_column: str,
  id_column: str,
) -> dict[str, Any]:
  files = [
    {
      "path": path.relative_to(source_root).as_posix(),
      "size": path.stat().st_size,
      "mtime_ns": path.stat().st_mtime_ns,
    }
    for path in parquet_files
  ]
  payload = {
    "kind": "fineweb_tokenizer_corpus",
    "corpus_name": corpus_name,
    "source_root": str(source_root),
    "separator": SOURCE_CORPUS_SEPARATOR,
    "document_format": "text+separator",
    "split": {
      "strategy": "document_hash",
      "seed": split_seed,
      "train": train_ratio,
      "val": val_ratio,
      "test": 1.0 - train_ratio - val_ratio,
      "included": "train",
    },
    "columns": {
      "text": text_column,
      "id": id_column,
    },
    "document_separator": SOURCE_CORPUS_SEPARATOR,
    "token_shard_size": _FINEWEB_TOKEN_SHARD_SIZE,
    "files": files,
  }
  fingerprint = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  ).hexdigest()[:16]
  return {**payload, "fingerprint": fingerprint}


def _unitoken_tokenizer_manifest(
  *,
  tokenizer_name: str,
  vocab_size: int,
  special_token_count: int,
  corpus_fingerprint: str,
) -> dict[str, Any]:
  payload = {
    "kind": "unitoken",
    "name": tokenizer_name,
    "vocab_size": vocab_size,
    "special_token_count": special_token_count,
    "corpus_fingerprint": corpus_fingerprint,
  }
  fingerprint = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  ).hexdigest()[:16]
  return {**payload, "fingerprint": fingerprint}


def _save_unitoken_tokenizer(
  trainer,
  tokenizer_path: Path,
  manifest: dict[str, Any] | None = None,
) -> None:
  tokenizer_path.mkdir(parents=True, exist_ok=True)
  trainer.save(tokenizer_path.name, outdir=tokenizer_path)
  old_vocab_path = tokenizer_path / f"vocab.{tokenizer_path.name}[u8].json"
  old_merges_path = tokenizer_path / f"merges.{tokenizer_path.name}[u8].txt"
  os.replace(old_vocab_path, tokenizer_path / "vocab.json")
  os.replace(old_merges_path, tokenizer_path / "merges.txt")
  payload = manifest or {
    "kind": "unitoken",
    "name": tokenizer_path.name,
  }
  _write_json(
    tokenizer_path / "manifest.json",
    {
      **payload,
      "vocab_file": "vocab.json",
      "merges_file": "merges.txt",
      "created_at": datetime.now(UTC).isoformat(),
    },
  )


class _NpyTokenShardWriter:
  def __init__(self, split_dir: Path, split_name: str, shard_size: int) -> None:
    self.split_dir = split_dir
    self.split_name = split_name
    self.shard_size = shard_size
    self.paths: list[Path] = []
    self.current_path: Path | None = None
    self.current_tmp_path: Path | None = None
    self.current_array: np.memmap | None = None
    self.offset = 0

  def append(self, tokens: list[int]) -> None:
    array = np.asarray(tokens, dtype=np.dtype(_TOKEN_CACHE_DTYPE))
    position = 0
    while position < len(array):
      if self.current_array is None:
        self._open_shard()
      assert self.current_array is not None
      remaining = self.shard_size - self.offset
      count = min(remaining, len(array) - position)
      self.current_array[self.offset : self.offset + count] = array[
        position : position + count
      ]
      self.offset += count
      position += count
      if self.offset == self.shard_size:
        self._finish_full_shard()

  def close(self) -> list[Path]:
    if self.current_array is not None:
      self._finish_partial_shard()
    elif not self.paths:
      self.split_dir.mkdir(parents=True, exist_ok=True)
      path = self._next_path()
      tmp_path = self._tmp_path(path)
      with tmp_path.open("wb") as handle:
        np.save(handle, np.asarray([], dtype=np.dtype(_TOKEN_CACHE_DTYPE)))
      os.replace(tmp_path, path)
      self.paths.append(path)
    return self.paths

  def _open_shard(self) -> None:
    self.split_dir.mkdir(parents=True, exist_ok=True)
    path = self._next_path()
    tmp_path = self._tmp_path(path)
    self.current_path = path
    self.current_tmp_path = tmp_path
    self.current_array = np.lib.format.open_memmap(
      tmp_path,
      mode="w+",
      dtype=np.dtype(_TOKEN_CACHE_DTYPE),
      shape=(self.shard_size,),
    )
    self.offset = 0

  def _finish_full_shard(self) -> None:
    assert self.current_array is not None
    assert self.current_path is not None
    assert self.current_tmp_path is not None
    self.current_array.flush()
    del self.current_array
    os.replace(self.current_tmp_path, self.current_path)
    self.paths.append(self.current_path)
    self.current_path = None
    self.current_tmp_path = None
    self.current_array = None
    self.offset = 0

  def _finish_partial_shard(self) -> None:
    assert self.current_array is not None
    assert self.current_path is not None
    assert self.current_tmp_path is not None
    final_tmp_path = self._final_tmp_path(self.current_path)
    final_array = np.lib.format.open_memmap(
      final_tmp_path,
      mode="w+",
      dtype=np.dtype(_TOKEN_CACHE_DTYPE),
      shape=(self.offset,),
    )
    if self.offset:
      final_array[:] = self.current_array[: self.offset]
    final_array.flush()
    self.current_array.flush()
    del final_array
    del self.current_array
    self.current_tmp_path.unlink(missing_ok=True)
    os.replace(final_tmp_path, self.current_path)
    self.paths.append(self.current_path)
    self.current_path = None
    self.current_tmp_path = None
    self.current_array = None
    self.offset = 0

  def _next_path(self) -> Path:
    return self.split_dir / f"{self.split_name}-{len(self.paths):06d}.npy"

  @staticmethod
  def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")

  @staticmethod
  def _final_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.final.tmp")


def _write_fineweb_parquet_token_shards(
  *,
  parquet_files: list[Path],
  source_root: Path,
  encoding_name: str,
  split_dirs: dict[str, Path],
  files_path: Path | None,
  train_ratio: float,
  val_ratio: float,
  split_seed: int,
  text_column: str,
  id_column: str,
  parquet_batch_size: int,
  shard_size: int,
) -> dict[str, dict[str, Any]]:
  split_metadata: dict[str, dict[str, int | list[str]]] = {
    "train": {
      "tokens_files": [],
      "token_count": 0,
      "file_count": 0,
      "byte_count": 0,
    },
    "val": {
      "tokens_files": [],
      "token_count": 0,
      "file_count": 0,
      "byte_count": 0,
    },
    "test": {
      "tokens_files": [],
      "token_count": 0,
      "file_count": 0,
      "byte_count": 0,
    },
  }
  writers = {
    name: _NpyTokenShardWriter(
      split_dir=split_dirs[name],
      split_name=name,
      shard_size=shard_size,
    )
    for name in split_metadata
  }
  encoding = get_tokenizer(encoding_name)
  phase = "write"
  record_count = 0
  files_handle = (
    files_path.open("w", encoding="utf-8") if files_path is not None else None
  )
  completed = False
  try:
    for records in _iter_fineweb_parquet_record_batches(
      parquet_files=parquet_files,
      source_root=source_root,
      text_column=text_column,
      id_column=id_column,
      batch_size=parquet_batch_size,
    ):
      split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
      }
      for record in records:
        split_name = _assign_file_split(
          record["split_key"],
          train_ratio=train_ratio,
          val_ratio=val_ratio,
          split_seed=split_seed,
        )
        split_records[split_name].append(record)
      for split_name, records_for_split in split_records.items():
        if not records_for_split:
          continue
        encoded_batch = encoding.encode_ordinary_batch(
          [record["text"] + SOURCE_CORPUS_SEPARATOR for record in records_for_split]
        )
        for record, tokens in zip(records_for_split, encoded_batch, strict=True):
          token_count = len(tokens)
          metadata = split_metadata[split_name]
          metadata["token_count"] = int(metadata["token_count"]) + token_count
          metadata["file_count"] = int(metadata["file_count"]) + 1
          metadata["byte_count"] = int(metadata["byte_count"]) + int(
            record["byte_count"]
          )
          record_count += 1
          if record_count % 100_000 == 0:
            total_tokens = sum(
              int(split["token_count"]) for split in split_metadata.values()
            )
            total_bytes = sum(
              int(split["byte_count"]) for split in split_metadata.values()
            )
            print(
              "fineweb parquet "
              f"{phase}: records={record_count} "
              f"tokens={total_tokens} bytes={total_bytes}",
              file=sys.stderr,
              flush=True,
            )
          writers[split_name].append(tokens)
          if files_handle is not None:
            files_handle.write(
              json.dumps(
                {
                  "id": record["id"],
                  "source": record["source"],
                  "row": record["row"],
                  "split": split_name,
                  "byte_count": record["byte_count"],
                  "token_count": token_count,
                },
                sort_keys=True,
              )
              + "\n"
            )
    completed = True
  finally:
    if files_handle is not None:
      files_handle.close()
    if completed:
      for split_name, writer in writers.items():
        split_metadata[split_name]["tokens_files"] = [
          path.relative_to(files_path.parent).as_posix()
          if files_path is not None
          else path.name
          for path in writer.close()
        ]
  return split_metadata


def _prepare_fineweb_split_dirs(split_dirs: dict[str, Path]) -> None:
  for split_dir in split_dirs.values():
    split_dir.mkdir(parents=True, exist_ok=True)
    for path in split_dir.iterdir():
      if path.is_file() and (
        path.suffix == ".npy" or path.name.endswith((".npy.tmp", ".npy.final.tmp"))
      ):
        path.unlink()


def _iter_fineweb_parquet_records(
  *,
  parquet_files: list[Path],
  source_root: Path,
  text_column: str,
  id_column: str,
  batch_size: int,
):
  for records in _iter_fineweb_parquet_record_batches(
    parquet_files=parquet_files,
    source_root=source_root,
    text_column=text_column,
    id_column=id_column,
    batch_size=batch_size,
  ):
    yield from records


def _iter_fineweb_parquet_record_batches(
  *,
  parquet_files: list[Path],
  source_root: Path,
  text_column: str,
  id_column: str,
  batch_size: int,
):
  try:
    import pyarrow.parquet as pq
  except ModuleNotFoundError as exc:
    raise ImportError("local FineWeb parquet publishing requires pyarrow") from exc

  for path in parquet_files:
    relative_path = path.relative_to(source_root).as_posix()
    schema = pq.read_schema(path)
    if text_column not in schema.names:
      raise ValueError(f"parquet file missing text column {text_column!r}: {path}")
    columns = [text_column]
    has_id = id_column in schema.names
    if has_id:
      columns.append(id_column)
    parquet_file = pq.ParquetFile(path)
    row_offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
      values = batch.to_pydict()
      texts = values[text_column]
      ids = values.get(id_column) if has_id else None
      records = []
      for index, text in enumerate(texts):
        if not text:
          continue
        row = row_offset + index
        doc_id = (
          str(ids[index])
          if ids is not None and ids[index]
          else f"{relative_path}:{row}"
        )
        text_value = str(text)
        records.append(
          {
            "id": doc_id,
            "split_key": doc_id,
            "source": relative_path,
            "row": row,
            "text": text_value,
            "byte_count": len(text_value.encode("utf-8")),
          }
        )
      row_offset += len(texts)
      if records:
        yield records


def _published_fineweb_parquet_digest(
  *,
  source_root: Path,
  parquet_files: list[Path],
  encoding_name: str,
  tokenizer_fingerprint: str,
  split_seed: int,
  train_ratio: float,
  val_ratio: float,
  test_ratio: float,
  text_column: str,
  id_column: str,
) -> str:
  files = [
    {
      "path": path.relative_to(source_root).as_posix(),
      "size": path.stat().st_size,
      "mtime_ns": path.stat().st_mtime_ns,
    }
    for path in parquet_files
  ]
  manifest = {
    "format_version": _PUBLISHED_DATASET_VERSION,
    "kind": "fineweb_parquet",
    "encoding_name": encoding_name,
    "tokenizer_fingerprint": tokenizer_fingerprint,
    "split": {
      "strategy": "document_hash",
      "seed": split_seed,
      "train": train_ratio,
      "val": val_ratio,
      "test": test_ratio,
    },
    "columns": {
      "text": text_column,
      "id": id_column,
    },
    "document_separator": SOURCE_CORPUS_SEPARATOR,
    "files": files,
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
  manifest_path = _dataset_manifest_path(config)
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
  tokens_file = split_metadata.get("tokens_file")
  tokens_files = split_metadata.get("tokens_files")
  if not isinstance(tokens_file, str) and not _is_string_list(tokens_files):
    raise ValueError(f"invalid dataset manifest: {manifest_path}")
  return manifest


def _load_dataset_token_arrays(
  config: DataConfig,
  manifest: dict[str, Any],
) -> list[np.ndarray]:
  manifest_path = _dataset_manifest_path(config)
  split_metadata = _split_metadata(manifest, config.split)
  token_paths = _split_token_paths(manifest_path.parent, split_metadata)
  token_arrays = [_load_token_array(path) for path in token_paths]
  token_count = sum(int(array.shape[0]) for array in token_arrays)
  if int(split_metadata["token_count"]) != token_count:
    raise ValueError(f"token count mismatch: {manifest_path}")
  return token_arrays


def _load_token_array(path: Path) -> np.ndarray:
  token_array = np.load(path, allow_pickle=False, mmap_mode="c")
  if token_array.dtype != np.dtype(_TOKEN_CACHE_DTYPE):
    raise ValueError(f"unsupported token dtype: {path}")
  if token_array.ndim != 1:
    raise ValueError(f"token dataset must be a 1D array: {path}")
  return token_array


def _tokenizer_fingerprint(encoding_name: str) -> str:
  if encoding_name.startswith("unitoken:"):
    tokenizer_path = Path(encoding_name.removeprefix("unitoken:"))
    manifest_path = tokenizer_path / "manifest.json"
    if manifest_path.exists():
      manifest = _read_json(manifest_path)
      fingerprint = manifest.get("fingerprint")
      if isinstance(fingerprint, str) and fingerprint:
        return fingerprint
  return encoding_name


def _split_token_paths(root: Path, split_metadata: dict[str, Any]) -> list[Path]:
  tokens_file = split_metadata.get("tokens_file")
  if isinstance(tokens_file, str):
    return [root / tokens_file]
  tokens_files = split_metadata.get("tokens_files")
  if _is_string_list(tokens_files):
    return [root / name for name in tokens_files]
  raise ValueError("dataset split does not define token files")


def _manifest_split_paths(
  dataset_root: Path,
  manifest: dict[str, Any],
) -> dict[str, Path | list[Path]]:
  return {
    name: paths[0] if len(paths) == 1 else paths
    for name, metadata in manifest["splits"].items()
    for paths in [_split_token_paths(dataset_root, metadata)]
  }


def _fineweb_parquet_shards_exist(
  manifest_path: Path,
  dataset_root: Path,
  version: str,
) -> bool:
  if not manifest_path.exists():
    return False
  manifest = _read_json(manifest_path)
  if manifest.get("version") != version:
    return False
  splits = manifest.get("splits")
  if not isinstance(splits, dict):
    return False
  try:
    paths = [
      path
      for metadata in splits.values()
      for path in _split_token_paths(dataset_root, metadata)
    ]
  except ValueError:
    return False
  return bool(paths) and all(path.exists() for path in paths)


def _flatten_split_paths(
  dataset_root: Path,
  metadata: dict[str, dict[str, Any]],
) -> list[Path]:
  paths: list[Path] = []
  for split_metadata in metadata.values():
    tokens_files = split_metadata.get("tokens_files")
    if _is_string_list(tokens_files):
      paths.extend(dataset_root / name for name in tokens_files)
  return paths


def _is_string_list(value: object) -> bool:
  return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _dataset_manifest_path(config: DataConfig) -> Path:
  if config.version == "latest":
    latest_path = config.dataset_root / "latest.json"
    latest = _read_json(latest_path)
    manifest = latest.get("manifest")
    if isinstance(manifest, str) and manifest:
      return config.dataset_root / manifest
  resolved_version = config.resolved_version or _resolve_dataset_version(
    config.dataset_root,
    config.version,
  )
  return config.dataset_root / "versions" / resolved_version / "manifest.json"


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
