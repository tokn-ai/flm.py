"""Benchmark repo BPE tokenizer training backends."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from flm_datasets.corpus import (
  SOURCE_CORPUS_SEPARATOR,
  SourceCorpusConfig,
  iter_source_files,
  write_source_corpus_file,
)
from flm_datasets.tokenizer import unitoken_special_tokens


@dataclass(frozen=True)
class TrainingResult:
  backend: str
  outdir: Path
  vocab_path: Path
  merges_path: Path
  timings: dict[str, float]

  @property
  def total_seconds(self) -> float:
    return sum(self.timings.values())


def main() -> None:
  args = _parse_args()
  corpus_config = SourceCorpusConfig(root=args.repo_root)
  source_files = iter_source_files(corpus_config)
  if args.max_files is not None:
    source_files = source_files[: args.max_files]
  special_tokens = unitoken_special_tokens(16)

  args.outdir.mkdir(parents=True, exist_ok=True)
  corpus_path = args.outdir / "repo_sources.txt"
  corpus_timings: dict[str, float] = {}
  _time(
    "write_corpus",
    corpus_timings,
    lambda: write_source_corpus_file(
      corpus_path,
      corpus_config,
      paths=source_files,
      separator=SOURCE_CORPUS_SEPARATOR,
    ),
  )
  unitoken_result = _train_unitoken(
    corpus_path=corpus_path,
    outdir=args.outdir / "unitoken",
    name=args.tokenizer_name,
    vocab_size=args.vocab_size,
    special_tokens=special_tokens,
  )
  hf_result = _train_hf(
    corpus_path=corpus_path,
    outdir=args.outdir / "hf",
    name=args.tokenizer_name,
    vocab_size=args.vocab_size,
    special_tokens=special_tokens,
  )

  print(f"repo_root={args.repo_root.resolve()}")
  print(f"files={len(source_files)}")
  print(f"corpus_file={corpus_path}")
  for name, seconds in corpus_timings.items():
    print(f"{name}_s={seconds:.3f}")
  print(f"vocab_size={args.vocab_size}")
  print()
  for result in [unitoken_result, hf_result]:
    print(result.backend)
    for name, seconds in result.timings.items():
      print(f"  {name}_s={seconds:.3f}")
    print(f"  total_s={result.total_seconds:.3f}")
    print(f"  vocab_file={result.vocab_path}")
    print(f"  merges_file={result.merges_path}")
  print()
  _print_artifact_parity(unitoken_result, hf_result)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo-root", type=Path, default=Path("."))
  parser.add_argument("--outdir", type=Path, default=Path("cache/tokenizer_bench"))
  parser.add_argument("--tokenizer-name", default="repo_bench")
  parser.add_argument("--vocab-size", type=int, default=8192)
  parser.add_argument("--max-files", type=int)
  return parser.parse_args()


def _train_unitoken(
  *,
  corpus_path: Path,
  outdir: Path,
  name: str,
  vocab_size: int,
  special_tokens: list[str],
) -> TrainingResult:
  from uni_tokenizer import BpeTrainer, PreTokenizer

  _reset_dir(outdir)
  timings: dict[str, float] = {}
  pre_tokenizer = PreTokenizer(special_tokens=special_tokens)
  words: dict[str, int] = {}

  def collect_words() -> None:
    for word, count in pre_tokenizer.get_words_from_file(corpus_path).items():
      words[word] = words.get(word, 0) + int(count)

  _time("pretokenize", timings, collect_words)
  trainer = BpeTrainer(special_tokens)
  _time("add_words", timings, lambda: trainer.add_words(words))
  _time("train", timings, lambda: trainer.train(vocab_size=vocab_size))
  _time("save", timings, lambda: trainer.save(name, outdir=outdir))
  return TrainingResult(
    backend="unitoken",
    outdir=outdir,
    vocab_path=outdir / f"vocab.{name}[u8].json",
    merges_path=outdir / f"merges.{name}[u8].txt",
    timings=timings,
  )


def _train_hf(
  *,
  corpus_path: Path,
  outdir: Path,
  name: str,
  vocab_size: int,
  special_tokens: list[str],
) -> TrainingResult:
  from tokenizers import Tokenizer
  from tokenizers.models import BPE
  from tokenizers.pre_tokenizers import ByteLevel
  from tokenizers.trainers import BpeTrainer

  _reset_dir(outdir)
  timings: dict[str, float] = {}
  tokenizer = Tokenizer(BPE(unk_token=None))
  tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
  trainer = BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=special_tokens,
    initial_alphabet=ByteLevel.alphabet(),
  )

  _time("train", timings, lambda: tokenizer.train([corpus_path.as_posix()], trainer))
  raw_vocab_path = outdir / "vocab.json"
  raw_merges_path = outdir / "merges.txt"
  _time("save", timings, lambda: tokenizer.model.save(str(outdir)))
  vocab_path = outdir / f"vocab.{name}[u8].json"
  merges_path = outdir / f"merges.{name}[u8].txt"
  raw_vocab_path.replace(vocab_path)
  _write_hf_merges_with_counts(raw_merges_path, merges_path)
  raw_merges_path.unlink()
  return TrainingResult(
    backend="hf-tokenizers",
    outdir=outdir,
    vocab_path=vocab_path,
    merges_path=merges_path,
    timings=timings,
  )


def _write_hf_merges_with_counts(source: Path, target: Path) -> None:
  lines = source.read_text(encoding="utf-8").splitlines()
  with target.open("w", encoding="utf-8") as file:
    for line in lines:
      if not line or line.startswith("#"):
        continue
      left, right = line.split(" ")
      file.write(f"{left} {right} => 0\n")


def _print_artifact_parity(left: TrainingResult, right: TrainingResult) -> None:
  left_vocab = json.loads(left.vocab_path.read_text(encoding="utf-8"))
  right_vocab = json.loads(right.vocab_path.read_text(encoding="utf-8"))
  left_merges = _merge_pairs(left.merges_path)
  right_merges = _merge_pairs(right.merges_path)
  print("artifact parity")
  print(f"  vocab_equal={left_vocab == right_vocab}")
  print(f"  merge_pairs_equal={left_merges == right_merges}")
  print(f"  {left.backend}_vocab_size={len(left_vocab)}")
  print(f"  {right.backend}_vocab_size={len(right_vocab)}")
  print(f"  {left.backend}_merge_lines={len(left_merges)}")
  print(f"  {right.backend}_merge_lines={len(right_merges)}")


def _merge_pairs(path: Path) -> list[str]:
  pairs: list[str] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    pairs.append(line.split(" => ", maxsplit=1)[0])
  return pairs


def _reset_dir(path: Path) -> None:
  if path.exists():
    shutil.rmtree(path)
  path.mkdir(parents=True)


def _time(name: str, timings: dict[str, float], callback: Callable[[], None]) -> None:
  start = time.perf_counter()
  callback()
  timings[name] = time.perf_counter() - start


if __name__ == "__main__":
  main()
