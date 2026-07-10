"""Download the canonical pre-tokenized FineWeb-10B speedrun shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "kjj0/fineweb10B-gpt2"
DEFAULT_ROOT = Path("data/fineweb10B")
DEFAULT_TRAIN_SHARDS = 102


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
  parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
  parser.add_argument(
    "--train-shards",
    type=int,
    default=DEFAULT_TRAIN_SHARDS,
    help="number of training shards to fetch, starting at index 1",
  )
  parser.add_argument(
    "--run",
    action="store_true",
    help="perform downloads; without this flag, print the file plan",
  )
  return parser


def filenames(train_shards: int) -> tuple[str, ...]:
  if train_shards < 1:
    raise ValueError("train_shards must be positive")
  return ("fineweb_val_000000.bin",) + tuple(
    f"fineweb_train_{index:06d}.bin" for index in range(1, train_shards + 1)
  )


def main() -> None:
  args = build_parser().parse_args()
  planned = filenames(args.train_shards)
  if not args.run:
    print(f"repo: {args.repo_id}")
    print(f"destination: {args.root}")
    print(f"files: {len(planned)} (use --run to download)")
    return
  args.root.mkdir(parents=True, exist_ok=True)
  for filename in planned:
    print(f"downloading {filename}")
    hf_hub_download(
      repo_id=args.repo_id,
      filename=filename,
      local_dir=args.root,
    )


if __name__ == "__main__":
  main()
