"""Repository source corpus helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCE_SUFFIXES = frozenset(
  {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
  }
)
DEFAULT_SOURCE_NAMES = frozenset({".editorconfig", ".gitignore"})
DEFAULT_EXCLUDED_DIRS = frozenset(
  {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
  }
)


@dataclass(frozen=True)
class SourceCorpusConfig:
  root: Path
  suffixes: frozenset[str] = DEFAULT_SOURCE_SUFFIXES
  names: frozenset[str] = DEFAULT_SOURCE_NAMES
  excluded_dirs: frozenset[str] = DEFAULT_EXCLUDED_DIRS


def iter_source_files(config: SourceCorpusConfig) -> list[Path]:
  root = config.root.resolve()
  paths: list[Path] = []

  for path in root.rglob("*"):
    if not path.is_file():
      continue
    relative_parts = path.relative_to(root).parts
    if any(part in config.excluded_dirs for part in relative_parts[:-1]):
      continue
    if path.name in config.names or path.suffix in config.suffixes:
      paths.append(path)

  return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def read_source_corpus(config: SourceCorpusConfig) -> str:
  root = config.root.resolve()
  chunks: list[str] = []

  for path in iter_source_files(config):
    relative_path = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8")
    chunks.append(f"<|file:{relative_path}|>\n{text}\n")

  return "\n".join(chunks)
