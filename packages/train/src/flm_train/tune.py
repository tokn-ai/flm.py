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
  FilesSinkConfig,
  RunConfig,
  apply_overrides,
  config_to_plain,
  load_experiment_config,
  write_yaml,
)
from flm_train.runner import ExperimentRunner, generate_run_id, run_experiment
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
    default="memory",
    help="Profiler to run: memory, torch, nsys, all, or a comma-separated list.",
  )
  parser.add_argument(
    "--memory-trace",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record detailed torch.cuda.memory allocation history.",
  )
  parser.add_argument("--torch-trace", action="store_true")
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
  profilers = parse_profilers(args.profiler)
  if "memory" in profilers:
    run_torch_memory_profile(config, trace=args.memory_trace, log=print)
  if "torch" in profilers:
    run_torch_profile(config, export_trace=args.torch_trace, log=print)
  if "nsys" in profilers:
    run_nsys_profile(config, trace=args.nsys_trace, log=print)
    return


def parse_profilers(value: str) -> tuple[str, ...]:
  profilers = tuple(item.strip() for item in value.split(",") if item.strip())
  if profilers == ("all",):
    return ("memory", "torch", "nsys")
  supported = {"memory", "torch", "nsys"}
  unknown = sorted(set(profilers) - supported)
  if unknown:
    raise ValueError(f"unsupported profiler: {unknown}")
  if not profilers:
    raise ValueError("profiler must not be empty")
  return profilers


def prepare_tune_config(
  config: ExperimentConfig,
  *,
  include_eval: bool,
  include_rollout: bool,
  include_checkpoint: bool,
) -> ExperimentConfig:
  config = _with_tune_run_id(config)
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
    sinks=(FilesSinkConfig(),),
  )


def _with_tune_run_id(config: ExperimentConfig) -> ExperimentConfig:
  if config.run.id is not None:
    return config
  return ExperimentConfig(
    name=config.name,
    data=config.data,
    model=config.model,
    optimizer=config.optimizer,
    loop=config.loop,
    eval=config.eval,
    rollout=config.rollout,
    checkpoint=config.checkpoint,
    system_metrics=config.system_metrics,
    run=RunConfig(
      id=f"tune-{generate_run_id()}",
      name=config.run.name,
      group=config.run.group,
    ),
    secrets=config.secrets,
    output=config.output,
    sinks=config.sinks,
  )


def run_torch_profile(
  config: ExperimentConfig,
  *,
  export_trace: bool = False,
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

  table_path = tune_dir / "memory_table.txt"
  summary_path = tune_dir / "summary.json"
  trace_path = tune_dir / "trace.json"
  if export_trace:
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
        "trace": str(trace_path) if export_trace else None,
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


def run_torch_memory_profile(
  config: ExperimentConfig,
  *,
  trace: bool = True,
  log,
) -> Path:
  tune_dir = config.run_dir / "tune" / "memory"
  tune_dir.mkdir(parents=True, exist_ok=True)
  cuda_enabled = config.loop.device.startswith("cuda") and torch.cuda.is_available()

  if cuda_enabled:
    torch.cuda.memory.reset_peak_memory_stats()
    torch.cuda.memory.reset_accumulated_memory_stats()
    if trace:
      torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="all",
        clear_history=True,
      )
    before_stats = torch.cuda.memory.memory_stats_as_nested_dict()
  else:
    before_stats = {}

  try:
    run_experiment(config, log=log)
  finally:
    if cuda_enabled and trace:
      torch.cuda.memory._record_memory_history(enabled=None)

  if cuda_enabled:
    torch.cuda.synchronize()
    after_stats = torch.cuda.memory.memory_stats_as_nested_dict()
    memory_viz_path = tune_dir / "memory_snapshot.pickle" if trace else None
    if trace:
      torch.cuda.memory._dump_snapshot(str(memory_viz_path))
    (tune_dir / "memory_summary.txt").write_text(
      torch.cuda.memory_summary(device=config.loop.device),
      encoding="utf-8",
    )
    _write_json(
      tune_dir / "memory_snapshot.json",
      torch.cuda.memory.memory_snapshot(include_traces=trace),
    )
  else:
    after_stats = {}
    memory_viz_path = None

  _write_json(tune_dir / "memory_stats_before.json", before_stats)
  _write_json(tune_dir / "memory_stats_after.json", after_stats)
  _write_json(
    tune_dir / "summary.json",
    {
      "cuda_available": cuda_enabled,
      "device": config.loop.device,
      "memory_stats_after": str(tune_dir / "memory_stats_after.json"),
      "memory_stats_before": str(tune_dir / "memory_stats_before.json"),
      "memory_summary": str(tune_dir / "memory_summary.txt") if cuda_enabled else None,
      "memory_viz": "https://pytorch.org/memory_viz",
      "memory_viz_snapshot": str(memory_viz_path) if memory_viz_path else None,
      "memory_viz_snapshot_format": "pytorch_cuda_memory_viz_pickle"
      if memory_viz_path
      else None,
      "profiler": "memory",
      "run_dir": str(config.run_dir),
      "snapshot": str(tune_dir / "memory_snapshot.json") if cuda_enabled else None,
      "trace": trace if cuda_enabled else False,
    },
  )
  log(f"tune=memory dir={tune_dir}")
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
  tune_dir = (config.run_dir / "tune" / "nsys").resolve()
  tune_dir.mkdir(parents=True, exist_ok=True)
  config_path = tune_dir / "config.resolved.yaml"
  write_yaml(config_path, config_to_plain(config))
  command = build_nsys_command(
    nsys=nsys,
    config_path=config_path,
    output_prefix=tune_dir / "profile",
    trace=trace,
  )
  stdout_path = tune_dir / "stdout.log"
  stderr_path = tune_dir / "stderr.log"
  (tune_dir / "command.json").write_text(
    json.dumps(
      {
        "command": command,
        "run_dir": str(config.run_dir),
        "stderr": str(stderr_path),
        "stdout": str(stdout_path),
      },
      indent=2,
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )
  log(" ".join(command))
  with stdout_path.open("w", encoding="utf-8") as stdout:
    with stderr_path.open("w", encoding="utf-8") as stderr:
      subprocess.run(command, check=True, stderr=stderr, stdout=stdout)
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
    f"--output={output_prefix.resolve()}",
    f"--trace={trace}",
    sys.executable,
    "-m",
    "flm_train.cli",
    str(config_path.resolve()),
  ]


def _write_json(path: Path, payload: object) -> None:
  path.write_text(
    json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _profiler_sort_key(activities: list[ProfilerActivity]) -> str:
  if ProfilerActivity.CUDA in activities:
    return "self_cuda_memory_usage"
  return "self_cpu_memory_usage"


def main(argv: Sequence[str] | None = None) -> None:
  run_from_args(parse_args(argv))


if __name__ == "__main__":
  main()
