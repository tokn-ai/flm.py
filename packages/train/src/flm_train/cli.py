"""Command-line entry points for training experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from flm_train.config import (
  ExperimentOverrides,
  WorkspaceOverrides,
  apply_overrides,
  apply_workspace_overrides,
  load_experiment_config,
  load_workspace_config,
)
from flm_train.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("config", type=Path)
  parser.add_argument("--workspace-config", type=Path, default=None)
  parser.add_argument("--project", default=None)
  parser.add_argument("--code-dir", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--output-root", type=Path, default=None)
  parser.add_argument("--device", default=None)
  parser.add_argument("--steps", type=int, default=None)
  parser.add_argument("--root-dir", type=Path, default=None)
  parser.add_argument("--seed", type=int, default=None)
  return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  return build_parser().parse_args(argv)


def run_from_args(args: argparse.Namespace) -> None:
  workspace = load_workspace_config(args.workspace_config)
  workspace = apply_workspace_overrides(
    workspace,
    WorkspaceOverrides(
      project=args.project,
      code_dir=args.code_dir,
      work_dir=args.work_dir,
      output_root=args.output_root or args.root_dir,
    ),
  )
  config = load_experiment_config(_resolve_code_path(workspace.code_dir, args.config))
  config = apply_overrides(
    config,
    ExperimentOverrides(
      device=args.device,
      steps=args.steps,
      seed=args.seed,
    ),
  )
  run_experiment(config, workspace=workspace, log=print)


def _resolve_code_path(code_dir: Path, path: Path) -> Path:
  if path.is_absolute():
    return path
  return code_dir / path


def main(argv: Sequence[str] | None = None) -> None:
  run_from_args(parse_args(argv))


if __name__ == "__main__":
  main()
