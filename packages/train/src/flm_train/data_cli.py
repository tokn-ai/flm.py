"""Command-line entry points for dataset publishing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from flm_train.config import WorkspaceConfig, load_workspace_config
from flm_train.data import (
  publish_fineweb2_dataset,
  publish_fineweb_parquet_dataset,
  publish_repo_source_dataset,
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("--workspace-config", type=Path, default=None)
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
    default=Path("cache/repo_sources_cl100k"),
  )
  publish.add_argument("--encoding-name", default="cl100k_base")
  publish.add_argument("--unitoken-vocab-size", type=int)
  publish.add_argument("--unitoken-special-token-count", type=int, default=16)
  publish.add_argument("--tokenizer-root", type=Path, default=Path("tokenizers"))
  publish.add_argument("--tokenizer-name")
  publish.add_argument("--train-ratio", type=float, default=0.98)
  publish.add_argument("--val-ratio", type=float, default=0.01)
  publish.add_argument("--test-ratio", type=float, default=0.01)
  publish.add_argument("--split-seed", type=int, default=42)

  fineweb2 = subcommands.add_parser("fineweb2")
  fineweb2_subcommands = fineweb2.add_subparsers(
    dest="fineweb2_command",
    required=True,
  )
  fineweb2_publish = fineweb2_subcommands.add_parser("publish")
  fineweb2_publish.add_argument("--dataset-root", type=Path, required=True)
  fineweb2_publish.add_argument("--config-name", required=True)
  fineweb2_publish.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-2")
  fineweb2_publish.add_argument("--source-split", default="train")
  fineweb2_publish.add_argument("--encoding-name", default="cl100k_base")
  fineweb2_publish.add_argument("--max-train-bytes", type=int, default=50_000_000)
  fineweb2_publish.add_argument("--max-val-bytes", type=int, default=2_000_000)
  fineweb2_publish.add_argument("--max-test-bytes", type=int, default=2_000_000)
  fineweb2_publish.add_argument("--train-ratio", type=float, default=0.98)
  fineweb2_publish.add_argument("--val-ratio", type=float, default=0.01)
  fineweb2_publish.add_argument("--test-ratio", type=float, default=0.01)
  fineweb2_publish.add_argument("--split-seed", type=int, default=42)
  fineweb2_publish.add_argument("--text-column", default="text")

  fineweb = subcommands.add_parser("fineweb")
  fineweb_subcommands = fineweb.add_subparsers(
    dest="fineweb_command",
    required=True,
  )
  fineweb_publish = fineweb_subcommands.add_parser("publish-local")
  fineweb_publish.add_argument("--source-root", type=Path, required=True)
  fineweb_publish.add_argument(
    "--dataset-root",
    type=Path,
    default=Path("cache/fineweb_10bt_8192"),
  )
  fineweb_publish.add_argument("--encoding-name", default="cl100k_base")
  fineweb_publish.add_argument("--unitoken-vocab-size", type=int)
  fineweb_publish.add_argument("--unitoken-special-token-count", type=int, default=16)
  fineweb_publish.add_argument(
    "--tokenizer-root", type=Path, default=Path("tokenizers")
  )
  fineweb_publish.add_argument("--tokenizer-name")
  fineweb_publish.add_argument("--train-ratio", type=float, default=0.8)
  fineweb_publish.add_argument("--val-ratio", type=float, default=0.1)
  fineweb_publish.add_argument("--test-ratio", type=float, default=0.1)
  fineweb_publish.add_argument("--split-seed", type=int, default=42)
  fineweb_publish.add_argument("--text-column", default="text")
  fineweb_publish.add_argument("--id-column", default="id")
  fineweb_publish.add_argument("--parquet-batch-size", type=int, default=1024)

  return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  return build_parser().parse_args(argv)


def run_from_args(args: argparse.Namespace) -> None:
  workspace = load_workspace_config(args.workspace_config)
  if args.command == "repo-sources" and args.repo_sources_command == "publish":
    info = publish_repo_source_dataset(
      repo_root=_resolve_code_path(workspace, args.repo_root),
      dataset_root=_resolve_workspace_path(workspace, args.dataset_root),
      encoding_name=args.encoding_name,
      unitoken_vocab_size=args.unitoken_vocab_size,
      unitoken_special_token_count=args.unitoken_special_token_count,
      tokenizer_root=_resolve_workspace_path(workspace, args.tokenizer_root),
      tokenizer_name=args.tokenizer_name,
      train_ratio=args.train_ratio,
      val_ratio=args.val_ratio,
      test_ratio=args.test_ratio,
      split_seed=args.split_seed,
    )
    _print_published_info(info)
    return
  if args.command == "fineweb2" and args.fineweb2_command == "publish":
    info = publish_fineweb2_dataset(
      dataset_root=_resolve_workspace_path(workspace, args.dataset_root),
      config_name=args.config_name,
      encoding_name=args.encoding_name,
      dataset_name=args.dataset_name,
      source_split=args.source_split,
      max_train_bytes=args.max_train_bytes,
      max_val_bytes=args.max_val_bytes,
      max_test_bytes=args.max_test_bytes,
      train_ratio=args.train_ratio,
      val_ratio=args.val_ratio,
      test_ratio=args.test_ratio,
      split_seed=args.split_seed,
      text_column=args.text_column,
    )
    _print_published_info(info)
    return
  if args.command == "fineweb" and args.fineweb_command == "publish-local":
    info = publish_fineweb_parquet_dataset(
      source_root=_resolve_workspace_path(workspace, args.source_root),
      dataset_root=_resolve_workspace_path(workspace, args.dataset_root),
      encoding_name=args.encoding_name,
      unitoken_vocab_size=args.unitoken_vocab_size,
      unitoken_special_token_count=args.unitoken_special_token_count,
      tokenizer_root=_resolve_workspace_path(workspace, args.tokenizer_root),
      tokenizer_name=args.tokenizer_name,
      train_ratio=args.train_ratio,
      val_ratio=args.val_ratio,
      test_ratio=args.test_ratio,
      split_seed=args.split_seed,
      text_column=args.text_column,
      id_column=args.id_column,
      parquet_batch_size=args.parquet_batch_size,
    )
    _print_published_info(info)
    return
  raise ValueError(f"unsupported command: {args.command}")


def _resolve_code_path(workspace: WorkspaceConfig, path: Path) -> Path:
  if path.is_absolute():
    return path
  return workspace.code_root / path


def _resolve_workspace_path(workspace: WorkspaceConfig, path: Path) -> Path:
  if path.is_absolute():
    return path
  return workspace.workspace_root / path


def _print_published_info(info) -> None:
  print(f"dataset_root={info.dataset_root}")
  print(f"version={info.version}")
  print(f"tokens={info.token_count}")
  print(f"files={info.file_count}")
  print(f"bytes={info.byte_count}")
  print(f"unigram_entropy_nats_per_token={info.unigram_entropy_nats_per_token:.6f}")
  for split_name, split_info in info.splits.items():
    print(
      f"{split_name}_tokens={split_info['token_count']} "
      f"{split_name}_files={split_info['file_count']} "
      f"{split_name}_bytes={split_info['byte_count']}"
    )


def main(argv: Sequence[str] | None = None) -> None:
  run_from_args(parse_args(argv))
