"""Tokenizer helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import tiktoken

UNITOKEN_PREFIX = "unitoken:"


class EncodingLike(Protocol):
  name: str
  n_vocab: int

  def encode_ordinary(self, text: str) -> list[int]: ...

  def decode(self, tokens: list[int]) -> str: ...


def get_tokenizer(name: str = "cl100k_base") -> EncodingLike:
  if name.startswith(UNITOKEN_PREFIX):
    return _get_unitokenizer(name.removeprefix(UNITOKEN_PREFIX))
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


def _get_unitokenizer(model_path: str):
  from uni_tokenizer import Encoding

  path = Path(model_path)
  special_tokens = {
    token: index for index, token in enumerate(unitoken_special_tokens(16))
  }
  return Encoding.from_files(
    path.name,
    vocab_file=path.parent / f"vocab.{path.name}[u8].json",
    merges_file=path.parent / f"merges.{path.name}[u8].txt",
    special_tokens=special_tokens,
  )
