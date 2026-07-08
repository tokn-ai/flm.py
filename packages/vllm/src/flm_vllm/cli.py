"""Command line entry points for FLM vLLM helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from flm_train.runner import rebuild_rollout_summaries
from flm_train.types import RolloutPromptConfig

from flm_vllm.export import export_reference_checkpoint
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


def rollout_main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Run FLM rollout with vLLM.")
  parser.add_argument("model_dir", type=Path)
  parser.add_argument("--encoding", required=True)
  parser.add_argument("--prompt", action="append", default=[])
  parser.add_argument("--max-new-tokens", type=int, default=64)
  parser.add_argument("--step", type=int, default=0)
  parser.add_argument("--temperature", type=float, default=0.0)
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
