import pytest
import torch
from flm_llm import NanoGPTSpeedrunConfig, NanoGPTSpeedrunModel


def test_nanogpt_speedrun_model_returns_softcapped_logits_and_loss() -> None:
  model = NanoGPTSpeedrunModel(_config())
  input_ids = torch.randint(0, 32, (2, 8))
  targets = torch.randint(0, 32, (2, 8))

  logits, loss = model(input_ids, targets)

  assert logits is not None
  assert logits.shape == (2, 8, 32)
  assert logits.abs().max() <= 5.0
  assert loss is not None
  assert torch.isfinite(loss)


def test_nanogpt_speedrun_model_ties_embedding_and_head() -> None:
  model = NanoGPTSpeedrunModel(_config())

  assert model.token_embedding.weight is model.lm_head.weight


def test_nanogpt_speedrun_model_zero_initializes_residual_projections() -> None:
  model = NanoGPTSpeedrunModel(_config())

  for block in model.blocks:
    torch.testing.assert_close(
      block.attn.out.weight,
      torch.zeros_like(block.attn.out.weight),
    )
    torch.testing.assert_close(
      block.ffn.down.weight,
      torch.zeros_like(block.ffn.down.weight),
    )


def test_nanogpt_speedrun_model_uses_value_and_embedding_skips() -> None:
  model = NanoGPTSpeedrunModel(_config())

  assert model.embedding_skip_weights is not None
  assert model.embedding_skip_weights.shape == (2,)
  assert model.value_mix_logits is not None
  assert model.value_mix_logits.shape == (1,)


def test_nanogpt_speedrun_model_supports_chunked_loss_without_softcap() -> None:
  config = _config(
    logit_softcap=None,
    logit_sigmoid_scale=None,
    loss_backend="linear_cross_entropy",
    mtp_weights=(1.0,),
  )
  model = NanoGPTSpeedrunModel(config)
  input_ids = torch.randint(0, 32, (2, 8))
  targets = torch.randint(0, 32, (2, 8))

  logits, loss = model(input_ids, targets, return_logits=False)

  assert logits is None
  assert loss is not None
  assert torch.isfinite(loss)


def test_nanogpt_speedrun_model_validates_block_skip_endpoints() -> None:
  with pytest.raises(ValueError, match="both be set"):
    NanoGPTSpeedrunModel(_config(block_skip_from=0, block_skip_to=None))


def test_nanogpt_speedrun_model_wires_portable_token_features() -> None:
  model = NanoGPTSpeedrunModel(
    _config(
      bigram_vocab_size=101,
      bigram_dim=8,
      bigram_sign_table_rows=16,
      partial_key_offset_layers=(1,),
    )
  )
  input_ids = torch.randint(0, 32, (2, 8))

  logits, _ = model(input_ids)

  assert model.token_smear is not None
  assert model.bigram_embedding is not None
  assert model.bigram_injection_weights is not None
  assert logits is not None
  assert torch.isfinite(logits).all()


def test_nanogpt_speedrun_model_multi_token_loss_matches_offsets() -> None:
  model = NanoGPTSpeedrunModel(
    _config(
      logit_softcap=None,
      logit_sigmoid_scale=None,
      mtp_weights=(1.0, 0.5),
    )
  )
  logits = torch.randn(2, 4, 32)
  targets = torch.randint(0, 32, (2, 4))
  primary = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1),
    targets.flatten(),
    reduction="sum",
  )
  second = torch.nn.functional.cross_entropy(
    logits[:, :-1].flatten(0, 1),
    targets[:, 1:].flatten(),
    reduction="sum",
  )

  loss = model._multi_token_loss(logits, targets)

  torch.testing.assert_close(loss, (primary + 0.5 * second) / targets.numel())


def test_nanogpt_speedrun_model_eval_loss_uses_primary_target_only() -> None:
  model = NanoGPTSpeedrunModel(_config(mtp_weights=(1.0, 0.5, 0.25)))
  model.eval()
  logits = torch.randn(2, 4, 32)
  targets = torch.randint(0, 32, (2, 4))

  loss = model._multi_token_loss(logits, targets)
  expected = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1),
    targets.flatten(),
  )

  torch.testing.assert_close(loss, expected)


def test_nanogpt_speedrun_model_skips_configured_attention_layer() -> None:
  model = NanoGPTSpeedrunModel(_config(attention_free_layer=1))
  calls = []
  handle = model.blocks[1].attn.register_forward_hook(lambda *args: calls.append(True))

  model(torch.randint(0, 32, (2, 8)))

  handle.remove()
  assert calls == []


def test_nanogpt_speedrun_model_initializes_attention_gates_and_xsa() -> None:
  model = NanoGPTSpeedrunModel(_config())

  assert model.attention_gate_weights.shape == (2, 2, 12)
  assert model.xsa_alphas is not None
  torch.testing.assert_close(
    model.xsa_alphas,
    torch.zeros_like(model.xsa_alphas),
  )


def _config(**overrides) -> NanoGPTSpeedrunConfig:
  values = {
    "vocab_size": 32,
    "max_seq_len": 8,
    "d_model": 16,
    "n_layers": 2,
    "n_heads": 2,
    "d_ff": 32,
    "block_skip_from": None,
    "block_skip_to": None,
    "logit_softcap": 5.0,
    "logit_sigmoid_scale": 5.0,
  }
  values.update(overrides)
  return NanoGPTSpeedrunConfig(**values)
