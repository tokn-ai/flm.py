"""Compare byte and Unicode UniToken BPE models on Parquet text subsets."""

from __future__ import annotations

import argparse
import json
import math
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
  train_corpus = args.outdir / "train.txt"
  eval_corpus = args.outdir / "eval.txt"
  train_stats = write_text_subset(
    args.train_parquet,
    train_corpus,
    text_column=args.text_column,
    max_bytes=args.train_bytes,
    batch_size=args.batch_size,
  )
  eval_stats = write_text_subset(
    args.eval_parquet,
    eval_corpus,
    text_column=args.text_column,
    max_bytes=args.eval_bytes,
    batch_size=args.batch_size,
  )
  if not train_stats["rows"] or not eval_stats["rows"]:
    raise ValueError("training and evaluation subsets must both contain text")

  special_tokens = unitoken_special_tokens(args.special_token_count)
  pretokenizer = PreTokenizer(special_tokens=special_tokens)
  started = time.perf_counter()
  words = {
    word: int(count)
    for word, count in pretokenizer.get_words_from_file(train_corpus).items()
  }
  pretokenize_seconds = time.perf_counter() - started

  results = []
  eval_text = eval_corpus.read_text(encoding="utf-8")
  for unit in args.unit:
    results.append(
      train_and_evaluate(
        unit=unit,
        words=words,
        eval_text=eval_text,
        outdir=args.outdir,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        rounds=args.rounds,
      )
    )

  report = {
    "format": "unitoken",
    "vocab_size": args.vocab_size,
    "train": {**train_stats, "path": str(train_corpus)},
    "eval": {**eval_stats, "path": str(eval_corpus)},
    "word_types": len(words),
    "pretokenize_seconds": pretokenize_seconds,
    "results": results,
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
  parser.add_argument("--train-bytes", type=int, default=64 * 1024 * 1024)
  parser.add_argument("--eval-bytes", type=int, default=16 * 1024 * 1024)
  parser.add_argument("--text-column", default="text")
  parser.add_argument("--batch-size", type=int, default=2048)
  parser.add_argument("--special-token-count", type=int, default=16)
  parser.add_argument("--rounds", type=int, default=3)
  args = parser.parse_args()
  for name in ("vocab_size", "train_bytes", "eval_bytes", "batch_size", "rounds"):
    if getattr(args, name) < 1:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  return args


def write_text_subset(
  parquet_paths: list[Path],
  output_path: Path,
  *,
  text_column: str,
  max_bytes: int,
  batch_size: int,
) -> dict[str, int]:
  """Write at most ``max_bytes`` of UTF-8 text, preserving document boundaries."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  rows = 0
  text_bytes = 0
  characters = 0
  separator = f"\n{SOURCE_CORPUS_SEPARATOR}\n".encode()
  with output_path.open("wb") as output:
    for text in _iter_parquet_text(parquet_paths, text_column, batch_size):
      prefix = separator if rows else b""
      remaining = max_bytes - output.tell() - len(prefix)
      if remaining <= 0:
        break
      encoded = text.encode("utf-8")
      if len(encoded) > remaining:
        encoded = encoded[:remaining].decode("utf-8", errors="ignore").encode("utf-8")
      if not encoded:
        break
      output.write(prefix)
      output.write(encoded)
      rows += 1
      text_bytes += len(encoded)
      characters += len(encoded.decode("utf-8"))
      if len(encoded) < len(text.encode("utf-8")):
        break
  return {
    "rows": rows,
    "characters": characters,
    "text_bytes": text_bytes,
    "corpus_bytes": output_path.stat().st_size,
    "corpus_characters": len(output_path.read_text(encoding="utf-8")),
  }


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
  if not token_ids:
    return 0.0
  counts = Counter(token_ids)
  total = len(token_ids)
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
