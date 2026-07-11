"""Compare repo tokenizer parity and encode timing across backends."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from flm_datasets.corpus import SourceCorpusConfig, read_source_corpus
from flm_datasets.tokenizer import (
  get_tokenizer,
  repo_bpe_encoding_name,
  unitoken_special_tokens,
)


def main() -> None:
  args = _parse_args()
  tokenizer_path = args.tokenizer_root / args.tokenizer_name
  suffix = (
    "byte"
    if (tokenizer_path.parent / f"vocab.{tokenizer_path.name}[byte].json").exists()
    else "u8"
  )
  vocab_path = tokenizer_path.parent / f"vocab.{tokenizer_path.name}[{suffix}].json"
  merges_path = tokenizer_path.parent / f"merges.{tokenizer_path.name}[{suffix}].txt"

  vocab = _read_vocab(vocab_path)
  merges = _read_merges(merges_path)
  special_tokens = {
    token: index for index, token in enumerate(unitoken_special_tokens(16))
  }
  encodings = {
    backend: get_tokenizer(repo_bpe_encoding_name(tokenizer_path, backend=backend))
    for backend in ["unitoken", "tiktoken", "hf"]
  }

  corpus = read_source_corpus(SourceCorpusConfig(root=args.repo_root))
  if args.max_chars is not None:
    corpus = corpus[: args.max_chars]
  samples = [
    "hello world",
    "def token_123(): return x + 1\n",
    "  indented_name = snake_case(arg)\n",
    corpus,
  ]

  print(f"tokenizer={tokenizer_path}")
  print(f"vocab_file={vocab_path}")
  print(f"merges_file={merges_path}")
  print(f"repo_root={args.repo_root.resolve()}")
  print()
  _print_vocab_report(vocab, merges, special_tokens, encodings)
  print()
  _print_encode_parity(samples, encodings)
  print()
  _print_timing(corpus, args.rounds, encodings)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo-root", type=Path, default=Path("."))
  parser.add_argument("--tokenizer-root", type=Path, default=Path("tokenizers"))
  parser.add_argument("--tokenizer-name", default="repo_8192")
  parser.add_argument("--max-chars", type=int, default=1_000_000)
  parser.add_argument("--rounds", type=int, default=7)
  return parser.parse_args()


def _read_vocab(path: Path) -> dict[str, int]:
  return json.loads(path.read_text(encoding="utf-8"))


def _read_merges(path: Path) -> list[tuple[str, str, int]]:
  merges: list[tuple[str, str, int]] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    pair, count = line.split(" => ")
    left, right = pair.split(" ")
    merges.append((left, right, int(count)))
  return merges


def _print_vocab_report(
  vocab: dict[str, int],
  merges: list[tuple[str, str, int]],
  special_tokens: dict[str, int],
  encodings,
) -> None:
  ids = set(vocab.values())
  expected_merge_count = len(vocab) - len(special_tokens) - 256
  merge_counts = [count for _left, _right, count in merges]
  merged_tokens_in_vocab = all(left + right in vocab for left, right, _count in merges)
  merged_ids_in_order = all(
    vocab[left + right] == len(special_tokens) + 256 + index
    for index, (left, right, _count) in enumerate(merges)
  )
  print("vocab/merges")
  print(f"  vocab_size={len(vocab)}")
  print(f"  special_tokens={len(special_tokens)}")
  print(f"  merge_lines={len(merges)} expected={expected_merge_count}")
  print(f"  ids_contiguous={ids == set(range(len(vocab)))}")
  print(
    f"  merge_counts_descending={merge_counts == sorted(merge_counts, reverse=True)}"
  )
  print(f"  merged_tokens_in_vocab={merged_tokens_in_vocab}")
  print(f"  merged_ids_in_order={merged_ids_in_order}")
  for backend, encoding in encodings.items():
    print(f"  {backend}_n_vocab={encoding.n_vocab}")


def _print_encode_parity(samples: list[str], encodings) -> None:
  print("encode parity")
  baseline = encodings["unitoken"]
  for index, sample in enumerate(samples):
    baseline_ids = baseline.encode_ordinary(sample)
    parity = {
      backend: baseline_ids == encoding.encode_ordinary(sample)
      for backend, encoding in encodings.items()
      if backend != "unitoken"
    }
    parity_text = " ".join(
      f"unitoken=={backend}:{matches}" for backend, matches in parity.items()
    )
    print(
      f"  sample={index} chars={len(sample)} tokens={len(baseline_ids)} {parity_text}"
    )


def _print_timing(corpus: str, rounds: int, encodings) -> None:
  print("timing")
  encoders: list[tuple[str, Callable[[str], list[int]]]] = [
    (backend, encoding.encode_ordinary) for backend, encoding in encodings.items()
  ]
  for name, encode in encoders:
    tokens = encode(corpus)
    times: list[float] = []
    for _ in range(rounds):
      start = time.perf_counter()
      tokens = encode(corpus)
      times.append(time.perf_counter() - start)
    best = min(times)
    median = statistics.median(times)
    print(
      f"  {name}: chars={len(corpus)} tokens={len(tokens)} "
      f"best_ms={best * 1000:.2f} median_ms={median * 1000:.2f} "
      f"best_ktok_s={len(tokens) / best / 1000:.1f}"
    )


if __name__ == "__main__":
  main()
