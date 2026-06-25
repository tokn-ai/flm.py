"""Tiktoken helpers."""

from __future__ import annotations

import tiktoken


def get_tokenizer(name: str = "cl100k_base") -> tiktoken.Encoding:
  return tiktoken.get_encoding(name)


def encode_text(text: str, encoding_name: str = "cl100k_base") -> list[int]:
  encoding = get_tokenizer(encoding_name)
  return encoding.encode_ordinary(text)
