import pytest
from flm_datasets import CalcQA, CalcQAConfig


def test_calcqa_generates_deterministic_examples() -> None:
  config = CalcQAConfig(size=3, seed=7, min_value=1, max_value=3, max_depth=2)

  first = CalcQA(config)
  second = CalcQA(config)

  assert len(first) == 3
  assert first[0] == second[0]
  assert first[0].question == f"What is {first[0].expression}?"


def test_calcqa_answer_matches_expression() -> None:
  dataset = CalcQA(
    CalcQAConfig(
      size=10,
      seed=11,
      min_value=1,
      max_value=5,
      max_depth=3,
    )
  )

  for example in dataset:
    assert eval(example.expression, {"__builtins__": {}}, {}) == example.answer


def test_calcqa_can_generate_leaf_expressions() -> None:
  dataset = CalcQA(CalcQAConfig(size=1, seed=1, min_value=5, max_value=5, max_depth=0))

  assert dataset[0].expression == "5"
  assert dataset[0].answer == 5
  assert dataset[0].question == "What is 5?"


def test_calcqa_rejects_invalid_config() -> None:
  with pytest.raises(ValueError, match="min_value"):
    CalcQA(CalcQAConfig(min_value=2, max_value=1))

  with pytest.raises(ValueError, match="operators"):
    CalcQA(CalcQAConfig(operators=()))
