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
  SOURCE_CORPUS_SEPARATOR,
  SourceCorpusConfig,
  TokenDataset,
  encode_text,
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
  split_paths: dict[str, Path]
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
  token_array = _load_dataset_tokens(resolved_data, metadata)
  dataset = TokenDataset(token_array, seq_len=resolved_data.seq_len)
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
  dataset_root: Path,
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
  if unitoken_vocab_size is not None:
    if unitoken_special_token_count != 16:
      raise ValueError("unitoken currently supports exactly 16 reserved special tokens")
    name = tokenizer_name or f"fineweb_10bt_{unitoken_vocab_size}"
    tokenizer_path = tokenizer_root / name
    train_unitoken_tokenizer_from_parquet(
      parquet_files=parquet_files,
      source_root=source_root,
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

  version = _published_fineweb_parquet_digest(
    source_root=source_root,
    parquet_files=parquet_files,
    encoding_name=encoding_name,
    split_seed=split_seed,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
    text_column=text_column,
    id_column=id_column,
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
    split_metadata = _scan_fineweb_parquet_tokens(
      parquet_files=parquet_files,
      source_root=source_root,
      encoding_name=encoding_name,
      split_paths=None,
      files_path=None,
      train_ratio=train_ratio,
      val_ratio=val_ratio,
      split_seed=split_seed,
      text_column=text_column,
      id_column=id_column,
      parquet_batch_size=parquet_batch_size,
    )
    version_dir.mkdir(parents=True, exist_ok=True)
    split_arrays = {
      name: np.lib.format.open_memmap(
        split_paths[name],
        mode="w+",
        dtype=np.dtype(_TOKEN_CACHE_DTYPE),
        shape=(int(metadata["token_count"]),),
      )
      for name, metadata in split_metadata.items()
    }
    written_metadata = _scan_fineweb_parquet_tokens(
      parquet_files=parquet_files,
      source_root=source_root,
      encoding_name=encoding_name,
      split_paths=split_arrays,
      files_path=files_path,
      train_ratio=train_ratio,
      val_ratio=val_ratio,
      split_seed=split_seed,
      text_column=text_column,
      id_column=id_column,
      parquet_batch_size=parquet_batch_size,
    )
    for array in split_arrays.values():
      array.flush()
    _write_json(
      manifest_path,
      {
        "version": version,
        "format_version": _PUBLISHED_DATASET_VERSION,
        "kind": "fineweb_parquet",
        "source_root": str(source_root),
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
  vocab_path = tokenizer_path.parent / f"vocab.{tokenizer_path.name}[u8].json"
  merges_path = tokenizer_path.parent / f"merges.{tokenizer_path.name}[u8].txt"
  if vocab_path.exists() and merges_path.exists():
    return

  from uni_tokenizer import BpeTrainer, PreTokenizer

  tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
  corpus_path = tokenizer_path.parent / f"corpus.{tokenizer_path.name}.txt"
  write_source_corpus_file(corpus_path, corpus_config, paths=source_files)
  pre_tokenizer = PreTokenizer(special_tokens=special_tokens)
  words: dict[str, int] = {}
  for word, count in pre_tokenizer.get_words_from_file(corpus_path).items():
    words[word] = words.get(word, 0) + int(count)

  trainer = BpeTrainer(special_tokens)
  trainer.add_words(words)
  trainer.train(vocab_size=vocab_size)
  trainer.save(tokenizer_path.name, outdir=tokenizer_path.parent)


def train_unitoken_tokenizer_from_parquet(
  *,
  parquet_files: list[Path],
  source_root: Path,
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
  vocab_path = tokenizer_path.parent / f"vocab.{tokenizer_path.name}[u8].json"
  merges_path = tokenizer_path.parent / f"merges.{tokenizer_path.name}[u8].txt"
  if vocab_path.exists() and merges_path.exists():
    return

  from uni_tokenizer import BpeTrainer, PreTokenizer

  tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
  corpus_path = tokenizer_path.parent / f"corpus.{tokenizer_path.name}.txt"
  with corpus_path.open("w", encoding="utf-8") as corpus_file:
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
      corpus_file.write("\n")
      corpus_file.write(SOURCE_CORPUS_SEPARATOR)
      corpus_file.write("\n")

  pre_tokenizer = PreTokenizer(special_tokens=special_tokens)
  words = {
    word: int(count)
    for word, count in pre_tokenizer.get_words_from_file(corpus_path).items()
  }
  trainer = BpeTrainer(special_tokens)
  trainer.add_words(words)
  trainer.train(vocab_size=vocab_size)
  trainer.save(tokenizer_path.name, outdir=tokenizer_path.parent)
  corpus_path.unlink(missing_ok=True)


def _scan_fineweb_parquet_tokens(
  *,
  parquet_files: list[Path],
  source_root: Path,
  encoding_name: str,
  split_paths: dict[str, np.ndarray] | None,
  files_path: Path | None,
  train_ratio: float,
  val_ratio: float,
  split_seed: int,
  text_column: str,
  id_column: str,
  parquet_batch_size: int,
) -> dict[str, dict[str, int | str]]:
  split_metadata: dict[str, dict[str, int | str]] = {
    "train": {
      "tokens_file": "train.npy",
      "token_count": 0,
      "file_count": 0,
      "byte_count": 0,
    },
    "val": {
      "tokens_file": "val.npy",
      "token_count": 0,
      "file_count": 0,
      "byte_count": 0,
    },
    "test": {
      "tokens_file": "test.npy",
      "token_count": 0,
      "file_count": 0,
      "byte_count": 0,
    },
  }
  offsets = {"train": 0, "val": 0, "test": 0}
  files_handle = (
    files_path.open("w", encoding="utf-8") if files_path is not None else None
  )
  try:
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
      tokens = encode_text(record["text"] + "\n\n", encoding_name=encoding_name)
      token_count = len(tokens)
      metadata = split_metadata[split_name]
      metadata["token_count"] = int(metadata["token_count"]) + token_count
      metadata["file_count"] = int(metadata["file_count"]) + 1
      metadata["byte_count"] = int(metadata["byte_count"]) + int(record["byte_count"])
      if split_paths is not None:
        offset = offsets[split_name]
        split_paths[split_name][offset : offset + token_count] = tokens
        offsets[split_name] = offset + token_count
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
  finally:
    if files_handle is not None:
      files_handle.close()
  return split_metadata


def _iter_fineweb_parquet_records(
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
        yield {
          "id": doc_id,
          "split_key": doc_id,
          "source": relative_path,
          "row": row,
          "text": text_value,
          "byte_count": len(text_value.encode("utf-8")),
        }
      row_offset += len(texts)


def _published_fineweb_parquet_digest(
  *,
  source_root: Path,
  parquet_files: list[Path],
  encoding_name: str,
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
  token_array = np.load(tokens_path, allow_pickle=False, mmap_mode="c")
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
