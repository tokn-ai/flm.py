import math

import torch
from flm_llm import ReferenceModel, ReferenceModelConfig


def test_reference_model_linear_cross_entropy_matches_cross_entropy() -> None:
  torch.manual_seed(0)
  input_ids = torch.randint(0, 32, (2, 8))
  targets = torch.randint(0, 32, (2, 8))
  standard = ReferenceModel(_config(loss_backend="cross_entropy"))
  chunked = ReferenceModel(_config(loss_backend="linear_cross_entropy"))
  chunked.load_state_dict(standard.state_dict())

  _, standard_loss = standard(input_ids, targets, return_logits=False)
  logits, chunked_loss = chunked(input_ids, targets, return_logits=False)

  assert logits is None
  assert standard_loss is not None
  assert chunked_loss is not None
  torch.testing.assert_close(chunked_loss, standard_loss)


def test_reference_model_returns_logits_for_inference() -> None:
  model = ReferenceModel(_config(loss_backend="linear_cross_entropy"))
  input_ids = torch.randint(0, 32, (2, 8))

  logits, loss = model(input_ids)

  assert logits is not None
  assert logits.shape == (2, 8, 32)
  assert loss is None


def test_reference_model_initial_embedding_scale_keeps_loss_near_uniform() -> None:
  torch.manual_seed(0)
  vocab_size = 8192
  config = ReferenceModelConfig(
    vocab_size=vocab_size,
    max_seq_len=64,
    d_model=256,
    n_layers=4,
    n_heads=16,
    d_ff=1024,
    loss_backend="cross_entropy",
  )
  model = ReferenceModel(config)
  input_ids = torch.randint(0, vocab_size, (2, 64))
  targets = torch.randint(0, vocab_size, (2, 64))

  _, loss = model(input_ids, targets, return_logits=False)

  assert loss is not None
  assert model.token_embedding.weight.min() >= -1.0 / config.d_model
  assert model.token_embedding.weight.max() <= 1.0 / config.d_model
  assert loss.item() < math.log(vocab_size) + 1.0


def _config(*, loss_backend: str) -> ReferenceModelConfig:
  return ReferenceModelConfig(
    vocab_size=32,
    max_seq_len=8,
    d_model=16,
    n_layers=1,
    n_heads=2,
    d_ff=16,
    loss_backend=loss_backend,
    loss_chunk_size=3,
  )
