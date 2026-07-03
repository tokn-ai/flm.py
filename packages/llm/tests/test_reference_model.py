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
