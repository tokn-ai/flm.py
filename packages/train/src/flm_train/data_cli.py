"""Command-line entry points for dataset publishing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from flm_train.data import publish_repo_source_dataset


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  subcommands = parser.add_subparsers(dest="command", required=True)

  repo_sources = subcommands.add_parser("repo-sources")
  repo_sources_subcommands = repo_sources.add_subparsers(
    dest="repo_sources_command",
    required=True,
  )
  publish = repo_sources_subcommands.add_parser("publish")
  publish.add_argument("--repo-root", type=Path, default=Path("."))
  publish.add_argument(
    "--dataset-root",
    type=Path,
    default=Path(".cache/data/repo_sources"),
  )
  publish.add_argument("--encoding-name", default="cl100k_base")
  publish.add_argument("--train-ratio", type=float, default=0.98)
  publish.add_argument("--val-ratio", type=float, default=0.01)
  publish.add_argument("--test-ratio", type=float, default=0.01)
  publish.add_argument("--split-seed", type=int, default=42)

  return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  return build_parser().parse_args(argv)


def run_from_args(args: argparse.Namespace) -> None:
  if args.command == "repo-sources" and args.repo_sources_command == "publish":
    info = publish_repo_source_dataset(
      repo_root=args.repo_root,
      dataset_root=args.dataset_root,
      encoding_name=args.encoding_name,
      train_ratio=args.train_ratio,
      val_ratio=args.val_ratio,
      test_ratio=args.test_ratio,
      split_seed=args.split_seed,
    )
    print(f"dataset_root={info.dataset_root}")
    print(f"version={info.version}")
    print(f"tokens={info.token_count}")
    print(f"files={info.file_count}")
    for split_name, split_info in info.splits.items():
      print(
        f"{split_name}_tokens={split_info['token_count']} "
        f"{split_name}_files={split_info['file_count']}"
      )
    return
  raise ValueError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
  run_from_args(parse_args(argv))
