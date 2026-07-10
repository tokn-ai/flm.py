#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

exec uv run flm-train-experiment \
  experiments/16m_fineweb_speedrun_smoke.yaml \
  "$@"
