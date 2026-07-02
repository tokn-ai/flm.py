"""Local secret environment loading."""

from __future__ import annotations

import os
from pathlib import Path


def load_secret_env(path: Path | None) -> dict[str, str]:
  if path is None or not path.exists():
    return {}

  values: dict[str, str] = {}
  lines = path.read_text(encoding="utf-8").splitlines()
  for line_number, raw_line in enumerate(lines, start=1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    if "=" not in line:
      raise ValueError(f"invalid secret line {line_number}: expected KEY=VALUE")
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
      raise ValueError(f"invalid secret line {line_number}: empty key")
    values[key] = _strip_quotes(value.strip())
  return values


def apply_secret_env(values: dict[str, str]) -> None:
  for key, value in values.items():
    os.environ.setdefault(key, value)


def _strip_quotes(value: str) -> str:
  if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
    return value[1:-1]
  return value
