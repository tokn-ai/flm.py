from flm_datasets import encode_text, get_tokenizer


def test_encode_text_uses_tiktoken() -> None:
  encoding = get_tokenizer("cl100k_base")

  tokens = encode_text("hello world", encoding_name="cl100k_base")

  assert tokens == encoding.encode_ordinary("hello world")
  assert tokens
