"""Synthetic arithmetic question-answer dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from torch.utils.data import Dataset

CalcOp = Literal["+", "-", "*"]


@dataclass(frozen=True)
class CalcQAConfig:
  size: int = 1_000
  seed: int = 42
  min_value: int = 0
  max_value: int = 20
  max_depth: int = 2
  operators: tuple[CalcOp, ...] = ("+", "-", "*")
  question_template: str = "What is {expression}?"


@dataclass(frozen=True)
class CalcQAExample:
  expression: str
  answer: int
  question: str


class CalcQA(Dataset[CalcQAExample]):
  """Random arithmetic expressions paired with exact integer answers."""

  def __init__(self, config: CalcQAConfig | None = None) -> None:
    self.config = config or CalcQAConfig()
    _validate_config(self.config)
    rng = random.Random(self.config.seed)
    self.examples = [_make_example(self.config, rng) for _ in range(self.config.size)]

  def __len__(self) -> int:
    return len(self.examples)

  def __getitem__(self, index: int) -> CalcQAExample:
    if index < 0 or index >= len(self):
      raise IndexError(index)
    return self.examples[index]


def _make_example(config: CalcQAConfig, rng: random.Random) -> CalcQAExample:
  expression, answer = _make_expression(config, rng, depth=config.max_depth)
  return CalcQAExample(
    expression=expression,
    answer=answer,
    question=config.question_template.format(expression=expression),
  )


def _make_expression(
  config: CalcQAConfig,
  rng: random.Random,
  depth: int,
) -> tuple[str, int]:
  if depth == 0:
    value = rng.randint(config.min_value, config.max_value)
    return str(value), value

  left_expr, left_value = _make_expression(config, rng, depth - 1)
  right_expr, right_value = _make_expression(config, rng, depth - 1)
  op = rng.choice(config.operators)
  value = _apply_op(left_value, right_value, op)
  return f"({left_expr} {op} {right_expr})", value


def _apply_op(left: int, right: int, op: CalcOp) -> int:
  if op == "+":
    return left + right
  if op == "-":
    return left - right
  if op == "*":
    return left * right
  raise ValueError(f"unsupported operator: {op}")


def _validate_config(config: CalcQAConfig) -> None:
  if config.size < 0:
    raise ValueError("size must be non-negative")
  if config.min_value > config.max_value:
    raise ValueError("min_value must be less than or equal to max_value")
  if config.max_depth < 0:
    raise ValueError("max_depth must be non-negative")
  if not config.operators:
    raise ValueError("operators must not be empty")
  unsupported = set(config.operators) - {"+", "-", "*"}
  if unsupported:
    raise ValueError(f"unsupported operators: {sorted(unsupported)}")
