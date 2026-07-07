"""Tokenizer helpers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import tiktoken

UNITOKEN_PREFIX = "unitoken:"
REPO_BPE_PREFIX = "repo_bpe:"
REPO_BPE_BACKEND_PREFIX = "repo_bpe+"
REPO_BPE_PAT_STR = (
  r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|"
  r"\s+(?!\S)|\s+"
)
RepoBpeBackend = Literal["unitoken", "tiktoken", "hf"]


class EncodingLike(Protocol):
  name: str
  n_vocab: int

  def encode_ordinary(self, text: str) -> list[int]: ...

  def decode(self, tokens: list[int]) -> str: ...


def get_tokenizer(name: str = "cl100k_base") -> EncodingLike:
  if name.startswith(UNITOKEN_PREFIX):
    return _get_unitokenizer(name.removeprefix(UNITOKEN_PREFIX))
  if name.startswith(REPO_BPE_PREFIX):
    return _get_repo_bpe_tokenizer(
      name.removeprefix(REPO_BPE_PREFIX),
      backend="unitoken",
    )
  if name.startswith(REPO_BPE_BACKEND_PREFIX):
    backend, model_path = _parse_repo_bpe_backend_name(name)
    return _get_repo_bpe_tokenizer(model_path, backend=backend)
  return tiktoken.get_encoding(name)


def encode_text(text: str, encoding_name: str = "cl100k_base") -> list[int]:
  if not text:
    return []
  encoding = get_tokenizer(encoding_name)
  return encoding.encode_ordinary(text)


def unitoken_special_tokens(count: int) -> list[str]:
  if count < 1:
    raise ValueError("unitoken special token count must be positive")
  return ["<|endoftext|>"] + [
    f"<|reserved_special_{index}|>" for index in range(1, count)
  ]


def unitoken_encoding_name(model_path: str | Path) -> str:
  return f"{UNITOKEN_PREFIX}{Path(model_path).as_posix()}"


def repo_bpe_encoding_name(
  model_path: str | Path,
  backend: RepoBpeBackend = "unitoken",
) -> str:
  path = Path(model_path).as_posix()
  if backend == "unitoken":
    return f"{REPO_BPE_PREFIX}{path}"
  return f"{REPO_BPE_BACKEND_PREFIX}{backend}:{path}"


def _get_unitokenizer(model_path: str):
  from uni_tokenizer import Encoding

  path = Path(model_path)
  special_tokens = {
    token: index for index, token in enumerate(unitoken_special_tokens(16))
  }
  vocab_file = path / "vocab.json"
  merges_file = path / "merges.txt"
  if not vocab_file.exists() or not merges_file.exists():
    vocab_file = path.parent / f"vocab.{path.name}[u8].json"
    merges_file = path.parent / f"merges.{path.name}[u8].txt"
  return Encoding.from_files(
    path.name,
    vocab_file=vocab_file,
    merges_file=merges_file,
    special_tokens=special_tokens,
  )


def _parse_repo_bpe_backend_name(name: str) -> tuple[RepoBpeBackend, str]:
  backend, separator, model_path = name.removeprefix(REPO_BPE_BACKEND_PREFIX).partition(
    ":"
  )
  if separator != ":" or not model_path:
    raise ValueError(f"invalid repo BPE encoding name: {name}")
  if backend not in {"unitoken", "tiktoken", "hf"}:
    raise ValueError(f"unsupported repo BPE backend: {backend}")
  return backend, model_path


def _get_repo_bpe_tokenizer(
  model_path: str,
  backend: RepoBpeBackend,
) -> EncodingLike:
  if backend == "unitoken":
    return _get_unitokenizer(model_path)
  if backend == "tiktoken":
    return _get_repo_bpe_tiktokenizer(model_path)
  if backend == "hf":
    return _get_repo_bpe_hf_tokenizer(model_path)
  raise ValueError(f"unsupported repo BPE backend: {backend}")


def _get_repo_bpe_tiktokenizer(model_path: str) -> tiktoken.Encoding:
  path = Path(model_path)
  vocab = _read_repo_bpe_vocab(path)
  special_tokens = _special_token_ranks()
  byte_decoder = {token: byte for byte, token in _bytes_to_unicode().items()}
  mergeable_ranks = {
    bytes(byte_decoder[char] for char in token): index
    for token, index in vocab.items()
    if token not in special_tokens
  }
  return tiktoken.Encoding(
    name=path.name,
    pat_str=REPO_BPE_PAT_STR,
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens,
    explicit_n_vocab=len(vocab),
  )


@dataclass
class HuggingFaceRepoBpeEncoding:
  name: str
  tokenizer: object

  @property
  def n_vocab(self) -> int:
    return self.tokenizer.get_vocab_size(with_added_tokens=True)

  def encode_ordinary(self, text: str) -> list[int]:
    return self.tokenizer.encode(text).ids

  def decode(self, tokens: list[int]) -> str:
    return self.tokenizer.decode(tokens)


def _get_repo_bpe_hf_tokenizer(model_path: str) -> HuggingFaceRepoBpeEncoding:
  from tokenizers import Tokenizer
  from tokenizers.decoders import ByteLevel as ByteLevelDecoder
  from tokenizers.models import BPE
  from tokenizers.pre_tokenizers import ByteLevel

  path = Path(model_path)
  vocab_path = _repo_bpe_vocab_path(path)
  merges = _read_repo_bpe_merges(path)
  with tempfile.NamedTemporaryFile(
    "w",
    encoding="utf-8",
    suffix=".txt",
    delete=False,
  ) as merges_file:
    merges_file.write("#version: 0.2\n")
    for left, right, _count in merges:
      merges_file.write(f"{left} {right}\n")
    hf_merges_path = merges_file.name
  try:
    tokenizer = Tokenizer(
      BPE.from_file(str(vocab_path), hf_merges_path, unk_token=None)
    )
  finally:
    os.unlink(hf_merges_path)
  tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
  tokenizer.decoder = ByteLevelDecoder()
  return HuggingFaceRepoBpeEncoding(path.name, tokenizer)


def _read_repo_bpe_vocab(path: Path) -> dict[str, int]:
  return json.loads(_repo_bpe_vocab_path(path).read_text(encoding="utf-8"))


def _read_repo_bpe_merges(path: Path) -> list[tuple[str, str, int]]:
  merges: list[tuple[str, str, int]] = []
  for line in _repo_bpe_merges_path(path).read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    pair, count = line.split(" => ")
    left, right = pair.split(" ")
    merges.append((left, right, int(count)))
  return merges


def _repo_bpe_vocab_path(path: Path) -> Path:
  return path.parent / f"vocab.{path.name}[u8].json"


def _repo_bpe_merges_path(path: Path) -> Path:
  return path.parent / f"merges.{path.name}[u8].txt"


def _special_token_ranks() -> dict[str, int]:
  return {token: index for index, token in enumerate(unitoken_special_tokens(16))}


def _bytes_to_unicode() -> dict[int, str]:
  byte_values = (
    list(range(ord("!"), ord("~") + 1))
    + list(range(ord("¡"), ord("¬") + 1))
    + list(range(ord("®"), ord("ÿ") + 1))
  )
  unicode_values = byte_values[:]
  counter = 0
  for byte in range(256):
    if byte not in byte_values:
      byte_values.append(byte)
      unicode_values.append(256 + counter)
      counter += 1
  return dict(zip(byte_values, map(chr, unicode_values), strict=True))
