from flm_datasets import encode_text, get_tokenizer
from flm_datasets.tokenizer import (
  repo_bpe_encoding_name,
  unitoken_encoding_name,
  unitoken_special_tokens,
)
from uni_tokenizer import BpeTrainer


def test_encode_text_uses_tiktoken() -> None:
  encoding = get_tokenizer("cl100k_base")

  tokens = encode_text("hello world", encoding_name="cl100k_base")

  assert tokens == encoding.encode_ordinary("hello world")
  assert tokens


def test_encode_text_supports_unitoken_files(tmp_path) -> None:
  tokenizer_path = _train_tiny_unitoken(tmp_path)
  special_tokens = unitoken_special_tokens(16)

  encoding_name = unitoken_encoding_name(tokenizer_path)
  encoding = get_tokenizer(encoding_name)
  tokens = encode_text("hello world", encoding_name=encoding_name)

  assert encoding.n_vocab <= 300
  assert tokens == encoding.encode_ordinary("hello world")
  assert encoding.decode(tokens) == "hello world"
  assert encoding.special_tokens_set == set(special_tokens)


def test_repo_bpe_backends_match_unitoken_files(tmp_path) -> None:
  tokenizer_path = _train_tiny_unitoken(tmp_path)
  samples = [
    "hello world",
    "def token_123(): return x + 1\n",
    "  indented_name = snake_case(arg)\n",
  ]

  encodings = [
    get_tokenizer(repo_bpe_encoding_name(tokenizer_path, backend=backend))
    for backend in ["unitoken", "tiktoken", "hf"]
  ]

  assert [encoding.n_vocab for encoding in encodings] == [300, 300, 300]
  for sample in samples:
    encoded = [encoding.encode_ordinary(sample) for encoding in encodings]
    assert encoded[0] == encoded[1] == encoded[2]
    assert encodings[0].decode(encoded[0]) == sample
    assert encodings[1].decode(encoded[1]) == sample
    assert encodings[2].decode(encoded[2]) == sample


def _train_tiny_unitoken(tmp_path):
  tokenizer_path = tmp_path / "tiny"
  special_tokens = unitoken_special_tokens(16)
  trainer = BpeTrainer(special_tokens)
  trainer.add_words(
    {
      "hello": 10,
      "world": 8,
      "hello world": 4,
      "def token_123(): return x + 1\n": 6,
      "  indented_name = snake_case(arg)\n": 6,
    }
  )
  trainer.train(vocab_size=300)
  trainer.save(tokenizer_path.name, outdir=tokenizer_path.parent)
  return tokenizer_path
