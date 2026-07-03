"""Command-line entry points for training experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from flm_train.config import (
  ExperimentOverrides,
  apply_overrides,
  load_experiment_config,
)
from flm_train.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("config", type=Path)
  parser.add_argument("--device", default=None)
  parser.add_argument("--steps", type=int, default=None)
  parser.add_argument("--root-dir", type=Path, default=None)
  parser.add_argument("--seed", type=int, default=None)
  return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  return build_parser().parse_args(argv)


def run_from_args(args: argparse.Namespace) -> None:
  config = load_experiment_config(args.config)
  config = apply_overrides(
    config,
    ExperimentOverrides(
      device=args.device,
      steps=args.steps,
      root_dir=args.root_dir,
      seed=args.seed,
    ),
  )
  run_experiment(config, log=print)


def main(argv: Sequence[str] | None = None) -> None:
  run_from_args(parse_args(argv))


if __name__ == "__main__":
  main()
