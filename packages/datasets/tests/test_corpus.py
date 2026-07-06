from pathlib import Path

from flm_datasets import (
  SOURCE_CORPUS_SEPARATOR,
  SourceCorpusConfig,
  iter_source_files,
  read_source_corpus,
  write_source_corpus_file,
)


def test_iter_source_files_uses_repo_source_suffixes(tmp_path: Path) -> None:
  (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
  (tmp_path / "b.toml").write_text("[x]\n", encoding="utf-8")
  (tmp_path / "ignored.bin").write_bytes(b"\x00")
  (tmp_path / ".venv").mkdir()
  (tmp_path / ".venv" / "site.py").write_text("x = 1\n", encoding="utf-8")
  (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
  (tmp_path / ".cache" / "data").mkdir(parents=True)
  (tmp_path / ".cache" / "data" / "tokens.json").write_text("[]\n", encoding="utf-8")

  paths = iter_source_files(SourceCorpusConfig(root=tmp_path))

  assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
    ".venv/site.py",
    "a.py",
    "b.toml",
  ]


def test_read_source_corpus_adds_file_markers(tmp_path: Path) -> None:
  (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")

  corpus = read_source_corpus(SourceCorpusConfig(root=tmp_path))

  assert "<|file:a.py|>" in corpus
  assert "x = 1" in corpus


def test_write_source_corpus_file_separates_files(tmp_path: Path) -> None:
  (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
  (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
  output_path = tmp_path / "corpus.txt"

  write_source_corpus_file(output_path, SourceCorpusConfig(root=tmp_path))

  corpus = output_path.read_text(encoding="utf-8")
  assert "<|file:a.py|>" in corpus
  assert f"\n{SOURCE_CORPUS_SEPARATOR}\n" in corpus
  assert "<|file:b.py|>" in corpus
