"""Diagnostic profiling entry points for training experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from flm_train.config import (
  ExperimentConfig,
  ExperimentOverrides,
  RunConfig,
  apply_overrides,
  config_to_plain,
  load_experiment_config,
  write_yaml,
)
from flm_train.runner import ExperimentRunner, run_experiment
from flm_train.types import CheckpointConfig


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("config", type=Path)
  parser.add_argument("--device", default=None)
  parser.add_argument("--steps", type=int, default=3)
  parser.add_argument("--root-dir", type=Path, default=None)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument(
    "--profiler",
    choices=("torch", "nsys", "both"),
    default="torch",
  )
  parser.add_argument(
    "--nsys-trace",
    default="cuda,nvtx,osrt,cudnn,cublas",
  )
  parser.add_argument("--include-eval", action="store_true")
  parser.add_argument("--include-rollout", action="store_true")
  parser.add_argument("--include-checkpoint", action="store_true")
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
  config = prepare_tune_config(
    config,
    include_eval=args.include_eval,
    include_rollout=args.include_rollout,
    include_checkpoint=args.include_checkpoint,
  )
  if args.profiler in {"torch", "both"}:
    run_torch_profile(config, log=print)
  if args.profiler in {"nsys", "both"}:
    run_nsys_profile(config, trace=args.nsys_trace, log=print)
    return


def prepare_tune_config(
  config: ExperimentConfig,
  *,
  include_eval: bool,
  include_rollout: bool,
  include_checkpoint: bool,
) -> ExperimentConfig:
  resolved = ExperimentRunner(config).resolved_config()
  return ExperimentConfig(
    name=resolved.name,
    data=resolved.data,
    model=resolved.model,
    optimizer=resolved.optimizer,
    loop=resolved.loop,
    eval=resolved.eval if include_eval else None,
    rollout=resolved.rollout if include_rollout else None,
    checkpoint=resolved.checkpoint
    if include_checkpoint
    else CheckpointConfig(enabled=False),
    system_metrics=replace(resolved.system_metrics, enabled=False),
    run=RunConfig(
      id=resolved.run.id,
      name=resolved.run.name,
      group=resolved.run.group,
    ),
    secrets=resolved.secrets,
    output=resolved.output,
    sinks=resolved.sinks,
  )


def run_torch_profile(
  config: ExperimentConfig,
  *,
  log,
) -> Path:
  tune_dir = config.run_dir / "tune" / "torch"
  tune_dir.mkdir(parents=True, exist_ok=True)
  activities = [ProfilerActivity.CPU]
  if config.loop.device.startswith("cuda") and torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

  with profile(
    activities=activities,
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
    with_modules=True,
  ) as profiler:
    run_experiment(config, log=log)

  trace_path = tune_dir / "trace.json"
  table_path = tune_dir / "memory_table.txt"
  summary_path = tune_dir / "summary.json"
  profiler.export_chrome_trace(str(trace_path))
  table_path.write_text(
    profiler.key_averages().table(
      sort_by=_profiler_sort_key(activities),
      row_limit=200,
    )
    + "\n",
    encoding="utf-8",
  )
  summary_path.write_text(
    json.dumps(
      {
        "profiler": "torch",
        "run_dir": str(config.run_dir),
        "trace": str(trace_path),
        "memory_table": str(table_path),
      },
      indent=2,
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )
  log(f"tune=torch dir={tune_dir}")
  return tune_dir


def run_nsys_profile(
  config: ExperimentConfig,
  *,
  trace: str,
  log,
) -> Path:
  nsys = shutil.which("nsys")
  if nsys is None:
    raise RuntimeError("nsys was not found on PATH")
  tune_dir = config.run_dir / "tune" / "nsys"
  tune_dir.mkdir(parents=True, exist_ok=True)
  config_path = tune_dir / "config.resolved.yaml"
  write_yaml(config_path, config_to_plain(config))
  command = build_nsys_command(
    nsys=nsys,
    config_path=config_path,
    output_prefix=tune_dir / "profile",
    trace=trace,
  )
  (tune_dir / "command.json").write_text(
    json.dumps(
      {
        "command": command,
        "run_dir": str(config.run_dir),
      },
      indent=2,
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )
  log(" ".join(command))
  subprocess.run(command, check=True)
  log(f"tune=nsys dir={tune_dir}")
  return tune_dir


def build_nsys_command(
  *,
  nsys: str,
  config_path: Path,
  output_prefix: Path,
  trace: str,
) -> list[str]:
  return [
    nsys,
    "profile",
    "--force-overwrite=true",
    f"--output={output_prefix}",
    f"--trace={trace}",
    sys.executable,
    "-m",
    "flm_train.cli",
    str(config_path),
  ]


def _profiler_sort_key(activities: list[ProfilerActivity]) -> str:
  if ProfilerActivity.CUDA in activities:
    return "self_cuda_memory_usage"
  return "self_cpu_memory_usage"


def main(argv: Sequence[str] | None = None) -> None:
  run_from_args(parse_args(argv))
