"""Compare byte and Unicode UniToken BPE models on Parquet text subsets."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from flm_datasets import SOURCE_CORPUS_SEPARATOR, unitoken_special_tokens
from uni_tokenizer import BpeEncoder, BpeTrainer, PreTokenizer

Unit = Literal["byte", "unicode"]
UNITS: tuple[Unit, ...] = ("byte", "unicode")


def main() -> None:
  args = _parse_args()
  args.outdir.mkdir(parents=True, exist_ok=True)
  metrics_path = args.outdir / "metrics.jsonl"
  metrics_path.write_text("")
  special_tokens = unitoken_special_tokens(args.special_token_count)
  models = []
  train_runs = []
  for train_bytes in args.train_bytes:
    trained, train_stats = train_models_streaming(
      parquet_paths=args.train_parquet,
      units=args.unit,
      outdir=args.outdir / _size_label(train_bytes),
      max_bytes=train_bytes,
      chunk_bytes=args.chunk_bytes,
      text_column=args.text_column,
      batch_size=args.batch_size,
      vocab_size=args.vocab_size,
      special_tokens=special_tokens,
      metrics_path=metrics_path,
    )
    models.extend(trained)
    train_runs.append(train_stats)

  eval_stats = evaluate_models_streaming(
    models,
    parquet_paths=args.eval_parquet,
    max_bytes=args.eval_bytes,
    chunk_bytes=args.chunk_bytes,
    text_column=args.text_column,
    batch_size=args.batch_size,
    metrics_path=metrics_path,
  )

  report = {
    "format": "unitoken",
    "vocab_size": args.vocab_size,
    "sources": {
      "train": [str(path) for path in args.train_parquet],
      "eval": [str(path) for path in args.eval_parquet],
    },
    "train_runs": train_runs,
    "eval": eval_stats,
    "results": models,
  }
  report_path = args.outdir / "results.json"
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
  print(json.dumps(report, ensure_ascii=False, indent=2))
  print(f"results_file={report_path}")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train-parquet", type=Path, nargs="+", required=True)
  parser.add_argument("--eval-parquet", type=Path, nargs="+", required=True)
  parser.add_argument("--outdir", type=Path, default=Path("cache/tokenizer_units"))
  parser.add_argument(
    "--unit",
    choices=UNITS,
    nargs="+",
    default=list(UNITS),
  )
  parser.add_argument("--vocab-size", type=int, default=8192)
  parser.add_argument(
    "--train-bytes",
    type=int,
    nargs="+",
    default=[1 * 1024**3, 10 * 1024**3],
  )
  parser.add_argument("--eval-bytes", type=int, default=100 * 1024**3)
  parser.add_argument("--chunk-bytes", type=int, default=64 * 1024**2)
  parser.add_argument("--text-column", default="text")
  parser.add_argument("--batch-size", type=int, default=2048)
  parser.add_argument("--special-token-count", type=int, default=16)
  args = parser.parse_args()
  for name in ("vocab_size", "eval_bytes", "chunk_bytes", "batch_size"):
    if getattr(args, name) < 1:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  if any(size < 1 for size in args.train_bytes):
    parser.error("--train-bytes values must be positive")
  return args


def train_models_streaming(
  *,
  parquet_paths: list[Path],
  units: list[Unit],
  outdir: Path,
  max_bytes: int,
  chunk_bytes: int,
  text_column: str,
  batch_size: int,
  vocab_size: int,
  special_tokens: list[str],
  metrics_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
  trainers = {
    unit: BpeTrainer(
      special_tokens,
      unit=unit,
      initial_alphabet="raw" if unit == "unicode" else None,
    )
    for unit in units
  }
  pretokenizer = PreTokenizer(special_tokens=special_tokens)
  alphabet: set[str] = set()
  total_word_count = 0
  stats = _empty_stream_stats(max_bytes)
  pretokenize_seconds = 0.0
  word_counts: Counter[str] = Counter()
  chunk_index = 0
  with tempfile.NamedTemporaryFile(suffix=".txt") as chunk_file:
    for chunk in iter_text_chunks(
      parquet_paths,
      max_bytes=max_bytes,
      chunk_bytes=chunk_bytes,
      text_column=text_column,
      batch_size=batch_size,
      stats=stats,
    ):
      chunk_file.seek(0)
      chunk_file.truncate()
      chunk_file.write(chunk.encode("utf-8"))
      chunk_file.flush()
      started = time.perf_counter()
      words = pretokenizer.get_words_from_file(chunk_file.name)
      chunk_seconds = time.perf_counter() - started
      pretokenize_seconds += chunk_seconds
      word_counts.update(words)
      if "unicode" in trainers:
        alphabet.update(character for word in words for character in word)
        total_word_count += sum(words.values())
      _append_metric(
        metrics_path,
        {
          "event": "train_chunk",
          "train_size": _size_label(max_bytes),
          "chunk": chunk_index,
          "corpus_bytes": stats["corpus_bytes"],
          "chunk_bytes": len(chunk.encode("utf-8")),
          "pretokenize_seconds": chunk_seconds,
          "word_types": len(words),
        },
      )
      chunk_index += 1
  if not stats["rows"]:
    raise ValueError("training stream did not contain text")
  add_seconds = {}
  for unit, trainer in trainers.items():
    training_words = word_counts
    if unit == "unicode":
      training_words = word_counts.copy()
      training_words.update({character: total_word_count + 1 for character in alphabet})
    started = time.perf_counter()
    trainer.add_words(training_words)
    add_seconds[unit] = time.perf_counter() - started

  models = []
  size_label = _size_label(max_bytes)
  for unit, trainer in trainers.items():
    model_name = f"cmn_{size_label}_{unit}_{vocab_size}"
    model_dir = outdir / unit
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trainer.train(vocab_size=vocab_size)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    trainer.save(model_name, outdir=model_dir, format="unitoken")
    save_seconds = time.perf_counter() - started
    _append_metric(
      metrics_path,
      {
        "event": "train_model",
        "train_size": size_label,
        "unit": unit,
        "add_words_seconds": add_seconds[unit],
        "train_seconds": train_seconds,
        "save_seconds": save_seconds,
        "word_types": len(word_counts),
        "alphabet_size": len(alphabet) if unit == "unicode" else 256,
      },
    )
    models.append(
      {
        "train_size": size_label,
        "train_bytes": max_bytes,
        "unit": unit,
        "format": "unitoken",
        "vocab_size": vocab_size,
        "alphabet_size": len(alphabet) if unit == "unicode" else 256,
        "model_name": model_name,
        "model_dir": str(model_dir),
        "add_words_seconds": add_seconds[unit],
        "train_seconds": train_seconds,
        "save_seconds": save_seconds,
        "encoder": BpeEncoder.load(
          model_name,
          unit=unit,
          format="unitoken",
          special_tokens=special_tokens,
          input_dir=model_dir,
        ),
      }
    )
  stats["pretokenize_seconds"] = pretokenize_seconds
  stats["size"] = size_label
  return models, stats


def evaluate_models_streaming(
  models: list[dict[str, object]],
  *,
  parquet_paths: list[Path],
  max_bytes: int,
  chunk_bytes: int,
  text_column: str,
  batch_size: int,
  metrics_path: Path,
) -> dict[str, object]:
  stats = _empty_stream_stats(max_bytes)
  states = [
    {"counts": Counter(), "tokens": 0, "seconds": 0.0, "roundtrip": True}
    for _ in models
  ]
  for chunk_index, chunk in enumerate(
    iter_text_chunks(
      parquet_paths,
      max_bytes=max_bytes,
      chunk_bytes=chunk_bytes,
      text_column=text_column,
      batch_size=batch_size,
      stats=stats,
    )
  ):
    chunk_metrics = []
    for model, state in zip(models, states, strict=True):
      encoder = model["encoder"]
      started = time.perf_counter()
      token_ids = encoder.encode(chunk)
      encode_seconds = time.perf_counter() - started
      state["seconds"] += encode_seconds
      state["tokens"] += len(token_ids)
      state["counts"].update(token_ids)
      state["roundtrip"] &= encoder.decode(token_ids) == chunk
      chunk_metrics.append(
        {
          "train_size": model["train_size"],
          "unit": model["unit"],
          "tokens": len(token_ids),
          "encode_seconds": encode_seconds,
        }
      )
    _append_metric(
      metrics_path,
      {
        "event": "eval_chunk",
        "chunk": chunk_index,
        "chunk_bytes": len(chunk.encode("utf-8")),
        "corpus_bytes": stats["corpus_bytes"],
        "models": chunk_metrics,
      },
    )
  if not stats["rows"]:
    raise ValueError("evaluation stream did not contain text")
  for model, state in zip(models, states, strict=True):
    model.pop("encoder")
    token_count = state["tokens"]
    entropy = entropy_from_counts(state["counts"], token_count)
    fixed_width_bits = math.ceil(math.log2(model["vocab_size"]))
    model.update(
      {
        "token_count": token_count,
        "unique_tokens": len(state["counts"]),
        "bytes_per_token": stats["corpus_bytes"] / token_count,
        "characters_per_token": stats["corpus_characters"] / token_count,
        "fixed_width_bits_per_token": fixed_width_bits,
        "fixed_width_compression_ratio": fixed_width_bits
        * token_count
        / (8 * stats["corpus_bytes"]),
        "entropy_bits_per_token": entropy,
        "entropy_bits_per_byte": entropy * token_count / stats["corpus_bytes"],
        "entropy_compression_ratio": entropy
        * token_count
        / (8 * stats["corpus_bytes"]),
        "encode_seconds": state["seconds"],
        "encode_mib_per_second": stats["corpus_bytes"] / 1024**2 / state["seconds"],
        "roundtrip": state["roundtrip"],
      }
    )
  return stats


def iter_text_chunks(
  parquet_paths: list[Path],
  *,
  max_bytes: int,
  chunk_bytes: int,
  text_column: str,
  batch_size: int,
  stats: dict[str, object],
) -> Iterator[str]:
  parts: list[str] = []
  buffered_bytes = 0
  separator = f"\n{SOURCE_CORPUS_SEPARATOR}\n"
  for text in _iter_parquet_text(parquet_paths, text_column, batch_size):
    prefix = separator if stats["rows"] else ""
    remaining = max_bytes - stats["corpus_bytes"]
    prefix_bytes = prefix.encode("utf-8")
    if remaining <= len(prefix_bytes):
      break
    text_bytes = text.encode("utf-8")[: remaining - len(prefix_bytes)]
    text_bytes = text_bytes.decode("utf-8", errors="ignore").encode("utf-8")
    if not text_bytes:
      break
    encoded = prefix_bytes + text_bytes
    value = prefix + text_bytes.decode("utf-8")
    parts.append(value)
    buffered_bytes += len(encoded)
    stats["rows"] += 1
    stats["corpus_bytes"] += len(encoded)
    stats["corpus_characters"] += len(value)
    if buffered_bytes >= chunk_bytes:
      yield "".join(parts)
      parts = []
      buffered_bytes = 0
    if stats["corpus_bytes"] >= max_bytes:
      break
  if parts:
    yield "".join(parts)


def _empty_stream_stats(max_bytes: int) -> dict[str, object]:
  return {
    "limit_bytes": max_bytes,
    "rows": 0,
    "corpus_bytes": 0,
    "corpus_characters": 0,
  }


def _size_label(size: int) -> str:
  if size % 1024**3 == 0:
    return f"{size // 1024**3}gib"
  if size % 1024**2 == 0:
    return f"{size // 1024**2}mib"
  return f"{size}b"


def _append_metric(path: Path, metric: dict[str, object]) -> None:
  with path.open("a", encoding="utf-8") as output:
    output.write(json.dumps(metric, ensure_ascii=False) + "\n")


def _iter_parquet_text(
  parquet_paths: list[Path], text_column: str, batch_size: int
) -> Iterator[str]:
  for parquet_path in parquet_paths:
    batches = pq.ParquetFile(parquet_path).iter_batches(
      columns=[text_column], batch_size=batch_size
    )
    for batch in batches:
      for text in batch.column(0).to_pylist():
        if text:
          yield str(text)


def train_and_evaluate(
  *,
  unit: Unit,
  words: dict[str, int],
  eval_text: str,
  outdir: Path,
  vocab_size: int,
  special_tokens: list[str],
  rounds: int,
) -> dict[str, object]:
  initial_alphabet = "raw" if unit == "unicode" else None
  training_words = words
  alphabet_size = 256
  if unit == "unicode":
    training_words, alphabet_size = reserve_unicode_alphabet(words)
  model_name = f"cmn_{unit}_{vocab_size}"
  model_dir = outdir / unit
  model_dir.mkdir(parents=True, exist_ok=True)
  trainer = BpeTrainer(
    special_tokens,
    unit=unit,
    initial_alphabet=initial_alphabet,
  )

  started = time.perf_counter()
  trainer.add_words(training_words)
  add_words_seconds = time.perf_counter() - started
  started = time.perf_counter()
  trainer.train(vocab_size=vocab_size)
  train_seconds = time.perf_counter() - started
  started = time.perf_counter()
  trainer.save(model_name, outdir=model_dir, format="unitoken")
  save_seconds = time.perf_counter() - started

  encoder = BpeEncoder.load(
    model_name,
    unit=unit,
    format="unitoken",
    special_tokens=special_tokens,
    input_dir=model_dir,
  )
  token_ids: list[int] = []
  encode_seconds = 0.0
  for _ in range(rounds):
    started = time.perf_counter()
    encoded = encoder.encode(eval_text)
    encode_seconds += time.perf_counter() - started
    token_ids = encoded
  started = time.perf_counter()
  decoded = encoder.decode(token_ids)
  decode_seconds = time.perf_counter() - started

  token_count = len(token_ids)
  utf8_bytes = len(eval_text.encode("utf-8"))
  characters = len(eval_text)
  entropy = empirical_entropy(token_ids)
  fixed_width_bits = math.ceil(math.log2(vocab_size))
  return {
    "unit": unit,
    "format": "unitoken",
    "initial_alphabet": initial_alphabet or "default",
    "alphabet_size": alphabet_size,
    "model_name": model_name,
    "model_dir": str(model_dir),
    "token_count": token_count,
    "unique_tokens": len(set(token_ids)),
    "bytes_per_token": utf8_bytes / token_count,
    "characters_per_token": characters / token_count,
    "tokens_per_character": token_count / characters,
    "fixed_width_bits_per_token": fixed_width_bits,
    "fixed_width_compression_ratio": (
      fixed_width_bits * token_count / (8 * utf8_bytes)
    ),
    "entropy_bits_per_token": entropy,
    "entropy_bits_per_byte": entropy * token_count / utf8_bytes,
    "entropy_bits_per_character": entropy * token_count / characters,
    "entropy_compression_ratio": entropy * token_count / (8 * utf8_bytes),
    "roundtrip": decoded == eval_text,
    "add_words_seconds": add_words_seconds,
    "train_seconds": train_seconds,
    "save_seconds": save_seconds,
    "encode_seconds_mean": encode_seconds / rounds,
    "encode_mib_per_second": (utf8_bytes / (1024 * 1024)) / (encode_seconds / rounds),
    "decode_seconds": decode_seconds,
  }


def empirical_entropy(token_ids: list[int]) -> float:
  """Return zero-order Shannon entropy in bits per token."""
  return entropy_from_counts(Counter(token_ids), len(token_ids))


def entropy_from_counts(counts: Counter[int], total: int) -> float:
  """Return zero-order Shannon entropy from an accumulated histogram."""
  if not total:
    return 0.0
  return -sum((count / total) * math.log2(count / total) for count in counts.values())


def reserve_unicode_alphabet(words: dict[str, int]) -> tuple[dict[str, int], int]:
  """Ensure every observed code point receives an atomic vocabulary entry."""
  alphabet = {character for word in words for character in word}
  priority = sum(words.values()) + 1
  prepared = words.copy()
  for character in alphabet:
    prepared[character] = prepared.get(character, 0) + priority
  return prepared, len(alphabet)


if __name__ == "__main__":
  main()
