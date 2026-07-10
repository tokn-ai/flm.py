"""Command line entry points for FLM vLLM helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from flm_train.runner import rebuild_rollout_summaries
from flm_train.types import RolloutPromptConfig

from flm_vllm.export import export_reference_checkpoint
from flm_vllm.importing import import_reference_export
from flm_vllm.rollout import generate_vllm_rollouts


def export_main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Export an FLM checkpoint for vLLM.")
  parser.add_argument("run_dir", type=Path)
  parser.add_argument("--checkpoint", default="latest")
  parser.add_argument("--output-dir", type=Path, default=None)
  args = parser.parse_args(argv)
  output_dir = export_reference_checkpoint(
    run_dir=args.run_dir,
    checkpoint=args.checkpoint,
    output_dir=args.output_dir,
  )
  print(output_dir)


def import_main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Validate and import an FLM vLLM export locally."
  )
  parser.add_argument("model_dir", type=Path)
  parser.add_argument("--map-location", default="cpu")
  args = parser.parse_args(argv)
  imported = import_reference_export(
    args.model_dir,
    map_location=args.map_location,
  )
  summary = {
    "model_dir": str(args.model_dir),
    "weight_path": str(imported.weight_path),
    "architecture": imported.config["architectures"][0],
    "vocab_size": imported.model.config.vocab_size,
    "max_seq_len": imported.model.config.max_seq_len,
    "d_model": imported.model.config.d_model,
    "n_layers": imported.model.config.n_layers,
    "n_heads": imported.model.config.n_heads,
  }
  print(json.dumps(summary, indent=2, sort_keys=True))


def rollout_main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Run FLM rollout with vLLM.")
  parser.add_argument("model_dir", type=Path)
  parser.add_argument("--encoding", default=None)
  parser.add_argument("--prompt", action="append", default=[])
  parser.add_argument("--max-new-tokens", type=int, default=64)
  parser.add_argument("--step", type=int, default=0)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--dtype", default="auto")
  parser.add_argument("--cpu-kvcache-space", type=int, default=None)
  parser.add_argument("--cpu-omp-threads-bind", default=None)
  parser.add_argument("--output-dir", type=Path, default=None)
  args = parser.parse_args(argv)
  prompts = tuple(_parse_prompt(value) for value in args.prompt)
  batch = generate_vllm_rollouts(
    model_dir=args.model_dir,
    encoding_name=args.encoding,
    prompts=prompts,
    max_new_tokens=args.max_new_tokens,
    step=args.step,
    temperature=args.temperature,
    dtype=args.dtype,
    cpu_kvcache_space=args.cpu_kvcache_space,
    cpu_omp_threads_bind=args.cpu_omp_threads_bind,
  )
  output_dir = args.output_dir or (args.model_dir / "rollouts")
  details_dir = output_dir / "details"
  details_dir.mkdir(parents=True, exist_ok=True)
  path = details_dir / f"step-{batch.step:08d}.json"
  path.write_text(
    json.dumps(asdict(batch), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  rebuild_rollout_summaries(output_dir)
  print(path)


def _parse_prompt(value: str) -> RolloutPromptConfig:
  if "=" not in value:
    raise ValueError("--prompt must be formatted as name=text")
  name, prompt = value.split("=", 1)
  return RolloutPromptConfig(name=name, prompt=prompt)
