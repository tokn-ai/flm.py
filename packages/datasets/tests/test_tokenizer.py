from flm_datasets import encode_text, get_tokenizer
from flm_datasets.tokenizer import unitoken_encoding_name, unitoken_special_tokens
from uni_tokenizer import BpeTrainer


def test_encode_text_uses_tiktoken() -> None:
  encoding = get_tokenizer("cl100k_base")

  tokens = encode_text("hello world", encoding_name="cl100k_base")

  assert tokens == encoding.encode_ordinary("hello world")
  assert tokens


def test_encode_text_supports_unitoken_files(tmp_path) -> None:
  tokenizer_path = tmp_path / "tiny"
  special_tokens = unitoken_special_tokens(16)
  trainer = BpeTrainer(special_tokens)
  trainer.add_words({"hello": 10, "world": 8, "hello world": 4})
  trainer.train(vocab_size=300)
  trainer.save(tokenizer_path.name, outdir=tokenizer_path.parent)

  encoding_name = unitoken_encoding_name(tokenizer_path)
  encoding = get_tokenizer(encoding_name)
  tokens = encode_text("hello world", encoding_name=encoding_name)

  assert encoding.n_vocab <= 300
  assert tokens == encoding.encode_ordinary("hello world")
  assert encoding.decode(tokens) == "hello world"
  assert encoding.special_tokens_set == set(special_tokens)
