import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_PATH = (
  Path(__file__).parents[3] / "scripts" / "experiment_parquet_tokenizer_units.py"
)
SPEC = importlib.util.spec_from_file_location("tokenizer_experiment", SCRIPT_PATH)
assert SPEC and SPEC.loader
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)
empirical_entropy = EXPERIMENT.empirical_entropy
reserve_unicode_alphabet = EXPERIMENT.reserve_unicode_alphabet
train_and_evaluate = EXPERIMENT.train_and_evaluate
iter_text_chunks = EXPERIMENT.iter_text_chunks


def test_empirical_entropy() -> None:
  assert empirical_entropy([]) == 0.0
  assert empirical_entropy([1, 1, 1, 1]) == 0.0
  assert empirical_entropy([1, 2, 1, 2]) == 1.0


def test_iter_text_chunks_respects_utf8_byte_limit(tmp_path: Path) -> None:
  parquet_path = tmp_path / "texts.parquet"
  pq.write_table(pa.table({"text": ["中文文本", "第二篇文档"]}), parquet_path)
  stats = {"rows": 0, "corpus_bytes": 0, "corpus_characters": 0}

  chunks = list(
    iter_text_chunks(
      [parquet_path],
      text_column="text",
      max_bytes=10,
      chunk_bytes=4,
      batch_size=2,
      stats=stats,
    )
  )

  assert chunks == ["中文文"]
  assert stats == {
    "rows": 1,
    "corpus_bytes": 9,
    "corpus_characters": 3,
  }


def test_reserve_unicode_alphabet_adds_singletons_without_changing_pairs() -> None:
  prepared, alphabet_size = reserve_unicode_alphabet({"中文": 3, "文档": 2})

  assert alphabet_size == 3
  assert prepared["中文"] == 3
  assert prepared["中"] > 3
  assert prepared["文"] > 3
  assert prepared["档"] > 3


def test_train_and_evaluate_byte_and_unicode(tmp_path: Path) -> None:
  text = "中文分词实验。你好，世界！"
  words = {text: 20, "中文": 10, "分词": 8} | {character: 100 for character in text}

  results = [
    train_and_evaluate(
      unit=unit,
      words=words,
      eval_text=text,
      outdir=tmp_path,
      vocab_size=270,
      special_tokens=["<|endoftext|>"],
      rounds=1,
    )
    for unit in ("byte", "unicode")
  ]

  assert [result["unit"] for result in results] == ["byte", "unicode"]
  assert all(result["roundtrip"] for result in results)
  assert all(result["token_count"] > 0 for result in results)
  assert (tmp_path / "byte" / "vocab.cmn_byte_270[byte].json").exists()
  assert (tmp_path / "unicode" / "vocab.cmn_unicode_270[unicode].json").exists()
