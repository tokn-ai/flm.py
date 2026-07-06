"""Mirror FineWeb/FineWeb2 language shards from Hugging Face.

This downloads raw parquet files. It is intentionally separate from
``flm-data fineweb2 publish``, which streams text and writes token datasets.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_ROOT = Path("/mnt/m/Corpus/FineWeb2")

FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB_REPO = "HuggingFaceFW/fineweb"

FINEWEB2_LANGUAGE_CONFIGS = {
  "fr": ("fra_Latn",),
  "de": ("deu_Latn",),
  "ru": ("rus_Cyrl",),
  "arb": ("arb_Arab",),
  "cjk": ("cmn_Hani", "jpn_Jpan", "kor_Hang"),
  "zh": ("cmn_Hani",),
  "ja": ("jpn_Jpan",),
  "ko": ("kor_Hang",),
}

DEFAULT_LANGUAGES = ("en", "cjk", "fr", "de", "ru", "arb")
DEFAULT_ENGLISH_DUMPS = ("CC-MAIN-2024-10",)
DEFAULT_ENGLISH_SAMPLES = ("10BT",)


@dataclass(frozen=True)
class DownloadPlan:
  repo_id: str
  local_dir: Path
  allow_patterns: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Download raw FineWeb/FineWeb2 parquet shards.",
  )
  parser.add_argument(
    "--root",
    type=Path,
    default=DEFAULT_ROOT,
    help=f"mirror root, default: {DEFAULT_ROOT}",
  )
  parser.add_argument(
    "--languages",
    nargs="+",
    default=DEFAULT_LANGUAGES,
    choices=tuple(sorted(("en", *FINEWEB2_LANGUAGE_CONFIGS.keys()))),
    help="language groups to download",
  )
  parser.add_argument(
    "--english-dump",
    action="append",
    dest="english_dumps",
    help=(
      "FineWeb dump for English, repeatable. Defaults to CC-MAIN-2024-10. "
      "Use --all-english-dumps to mirror every available FineWeb dump."
    ),
  )
  parser.add_argument(
    "--all-english-dumps",
    action="store_true",
    help="download every FineWeb English dump instead of selected dumps",
  )
  parser.add_argument(
    "--english-sample",
    action="append",
    dest="english_samples",
    help=(
      "FineWeb English sample size, repeatable. Defaults to 10BT. "
      "Use --no-english-samples to skip samples."
    ),
  )
  parser.add_argument(
    "--no-english-samples",
    action="store_true",
    help="skip FineWeb English sample downloads",
  )
  parser.add_argument(
    "--splits",
    nargs="+",
    default=("train", "test"),
    choices=("train", "test"),
    help="FineWeb2 splits to download",
  )
  parser.add_argument(
    "--max-workers",
    type=int,
    default=8,
    help="parallel download workers passed to huggingface_hub",
  )
  parser.add_argument(
    "--run",
    action="store_true",
    help="execute downloads. Without this flag, only print the plan.",
  )
  return parser


def build_plans(args: argparse.Namespace) -> list[DownloadPlan]:
  languages = tuple(dict.fromkeys(args.languages))
  plans: list[DownloadPlan] = []

  if "en" in languages:
    plans.append(
      DownloadPlan(
        repo_id=FINEWEB_REPO,
        local_dir=args.root / "fineweb",
        allow_patterns=_fineweb_allow_patterns(
          all_dumps=args.all_english_dumps,
          english_dumps=tuple(args.english_dumps or DEFAULT_ENGLISH_DUMPS),
          english_samples=()
          if args.no_english_samples
          else tuple(args.english_samples or DEFAULT_ENGLISH_SAMPLES),
        ),
      )
    )

  fineweb2_configs = tuple(
    dict.fromkeys(
      config
      for language in languages
      if language != "en"
      for config in FINEWEB2_LANGUAGE_CONFIGS[language]
    )
  )
  if fineweb2_configs:
    plans.append(
      DownloadPlan(
        repo_id=FINEWEB2_REPO,
        local_dir=args.root / "fineweb-2",
        allow_patterns=_fineweb2_allow_patterns(
          configs=fineweb2_configs,
          splits=tuple(args.splits),
        ),
      )
    )

  return plans


def _fineweb_allow_patterns(
  *,
  all_dumps: bool,
  english_dumps: tuple[str, ...],
  english_samples: tuple[str, ...],
) -> tuple[str, ...]:
  data_patterns = ("data/*",) if all_dumps else tuple(
    f"data/{dump}/*" for dump in english_dumps
  )
  sample_patterns = tuple(f"sample/{sample}/*" for sample in english_samples)
  return data_patterns + sample_patterns + ("README.md",)


def _fineweb2_allow_patterns(
  *,
  configs: tuple[str, ...],
  splits: tuple[str, ...],
) -> tuple[str, ...]:
  patterns = [
    f"data/{config}/{split}/*" for config in configs for split in splits
  ]
  return tuple(patterns) + ("README.md",)


def print_plan(plans: Sequence[DownloadPlan]) -> None:
  for plan in plans:
    print(f"repo={plan.repo_id}")
    print(f"local_dir={plan.local_dir}")
    print("allow_patterns:")
    for pattern in plan.allow_patterns:
      print(f"  {pattern}")
    print()


def run_downloads(plans: Sequence[DownloadPlan], *, max_workers: int) -> None:
  for plan in plans:
    print(f"Downloading {plan.repo_id} to {plan.local_dir}")
    snapshot_download(
      repo_id=plan.repo_id,
      repo_type="dataset",
      local_dir=plan.local_dir,
      allow_patterns=list(plan.allow_patterns),
      max_workers=max_workers,
    )


def main(argv: Sequence[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  plans = build_plans(args)
  print_plan(plans)
  if args.run:
    run_downloads(plans, max_workers=args.max_workers)
  else:
    print("Dry run only. Add --run to start downloads.")


if __name__ == "__main__":
  main()
