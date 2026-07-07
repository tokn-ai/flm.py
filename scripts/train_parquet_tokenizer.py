"""Train a Uni BPE tokenizer from one parquet text shard."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
from flm_datasets import SOURCE_CORPUS_SEPARATOR, unitoken_special_tokens
from uni_tokenizer import BpeTrainer, PreTokenizer


def main() -> None:
  args = _parse_args()
  tokenizer_root = args.tokenizer_root
  tokenizer_root.mkdir(parents=True, exist_ok=True)
  corpus_path = args.corpus_path or tokenizer_root / f"corpus.{args.tokenizer_name}.txt"
  vocab_path = tokenizer_root / f"vocab.{args.tokenizer_name}[u8].json"
  merges_path = tokenizer_root / f"merges.{args.tokenizer_name}[u8].txt"

  timings: dict[str, float] = {}
  rows, chars = _time_result(
    "write_corpus",
    timings,
    lambda: _write_parquet_text_corpus(
      parquet_path=args.parquet_path,
      corpus_path=corpus_path,
      text_column=args.text_column,
      batch_size=args.batch_size,
    ),
  )

  special_tokens = unitoken_special_tokens(args.special_token_count)
  pre_tokenizer = PreTokenizer(special_tokens=special_tokens)
  words: dict[str, int] = {}

  def collect_words() -> None:
    for word, count in pre_tokenizer.get_words_from_file(corpus_path).items():
      words[word] = int(count)

  _time("pretokenize", timings, collect_words)
  trainer = BpeTrainer(special_tokens)
  _time("add_words", timings, lambda: trainer.add_words(words))
  _time("train", timings, lambda: trainer.train(vocab_size=args.vocab_size))
  _time(
    "save",
    timings,
    lambda: trainer.save(args.tokenizer_name, outdir=tokenizer_root),
  )

  print(f"parquet={args.parquet_path}")
  print(f"text_column={args.text_column}")
  print(f"rows={rows}")
  print(f"chars={chars}")
  print(f"corpus_file={corpus_path}")
  print(f"corpus_bytes={corpus_path.stat().st_size}")
  print(f"vocab_size={args.vocab_size}")
  print(f"word_types={len(words)}")
  for name, seconds in timings.items():
    print(f"{name}_s={seconds:.3f}")
  print(f"total_s={sum(timings.values()):.3f}")
  print(f"vocab_file={vocab_path}")
  print(f"merges_file={merges_path}")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--parquet-path", type=Path, required=True)
  parser.add_argument("--tokenizer-root", type=Path, default=Path("tokenizers"))
  parser.add_argument("--tokenizer-name", required=True)
  parser.add_argument("--vocab-size", type=int, default=8192)
  parser.add_argument("--text-column", default="text")
  parser.add_argument("--batch-size", type=int, default=2048)
  parser.add_argument("--special-token-count", type=int, default=16)
  parser.add_argument("--corpus-path", type=Path)
  return parser.parse_args()


def _write_parquet_text_corpus(
  *,
  parquet_path: Path,
  corpus_path: Path,
  text_column: str,
  batch_size: int,
) -> tuple[int, int]:
  corpus_path.parent.mkdir(parents=True, exist_ok=True)
  parquet_file = pq.ParquetFile(parquet_path)
  rows = 0
  chars = 0
  with corpus_path.open("w", encoding="utf-8") as output:
    first = True
    batches = parquet_file.iter_batches(columns=[text_column], batch_size=batch_size)
    for batch in batches:
      for text in batch.column(0).to_pylist():
        if text is None:
          continue
        if not first:
          output.write(f"\n{SOURCE_CORPUS_SEPARATOR}\n")
        first = False
        output.write(text)
        output.write("\n")
        rows += 1
        chars += len(text)
  return rows, chars


def _time(name: str, timings: dict[str, float], callback: Callable[[], None]) -> None:
  start = time.perf_counter()
  callback()
  timings[name] = time.perf_counter() - start


def _time_result[T](
  name: str,
  timings: dict[str, float],
  callback: Callable[[], T],
) -> T:
  start = time.perf_counter()
  result = callback()
  timings[name] = time.perf_counter() - start
  return result


if __name__ == "__main__":
  main()
