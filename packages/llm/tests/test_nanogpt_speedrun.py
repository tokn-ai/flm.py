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

  assert model.embeddings_tied
  assert model.classifier_weight is model.lm_head.weight
  assert model.token_embedding.weight is not model.lm_head.weight
  torch.testing.assert_close(model.token_embedding.weight, model.lm_head.weight)

  model.untie_embeddings()

  assert not model.embeddings_tied
  assert model.classifier_weight is model.lm_head.weight


def test_nanogpt_speedrun_model_aggregates_tied_gradients_and_syncs() -> None:
  model = NanoGPTSpeedrunModel(_config())
  input_ids = torch.randint(0, 32, (2, 8))
  targets = torch.randint(0, 32, (2, 8))
  _, loss = model(input_ids, targets, return_logits=False)
  assert loss is not None
  loss.backward()
  embedding_grad = model.token_embedding.weight.grad.clone()
  head_grad = model.lm_head.weight.grad.clone()

  model.prepare_optimizer_step()

  assert model.token_embedding.weight.grad is None
  torch.testing.assert_close(
    model.lm_head.weight.grad,
    embedding_grad + head_grad,
  )
  with torch.no_grad():
    model.lm_head.weight.add_(1.0)
  model.finalize_optimizer_step()
  torch.testing.assert_close(model.token_embedding.weight, model.lm_head.weight)


def test_nanogpt_speedrun_model_matches_current_projection_initialization() -> None:
  model = NanoGPTSpeedrunModel(_config())

  for block in model.blocks:
    assert torch.count_nonzero(block.attn.out.weight) > 0
    torch.testing.assert_close(
      block.ffn.down.weight,
      torch.zeros_like(block.ffn.down.weight),
    )
  torch.testing.assert_close(
    model.attention_scales,
    torch.tensor((0.5, 1.0)).repeat(model.config.n_layers, 1),
  )


def test_nanogpt_speedrun_model_uses_current_embedding_skips() -> None:
  model = NanoGPTSpeedrunModel(_config())

  assert model.embedding_skip_weights is not None
  assert model.embedding_skip_weights.shape == (2,)
  assert model.value_mix_logits is None
  assert model.value_embeddings.shape == (1, 32, 16)
  assert model.value_gate_weights.shape == (1, 2, 12)
  torch.testing.assert_close(
    model.residual_scales,
    torch.full_like(model.residual_scales, model.config.residual_decay**0.5),
  )
  torch.testing.assert_close(model.post_scales, torch.ones_like(model.post_scales))


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
    logits.flatten(0, 1)[:-1],
    targets.flatten()[1:],
    reduction="sum",
  )

  loss = model._multi_token_loss(logits, targets)

  torch.testing.assert_close(loss, (primary + 0.5 * second) / targets.numel())


def test_nanogpt_speedrun_model_multi_token_loss_ignores_packed_padding() -> None:
  model = NanoGPTSpeedrunModel(
    _config(
      logit_softcap=None,
      logit_sigmoid_scale=None,
      mtp_weights=(1.0, 0.5),
    )
  )
  logits = torch.randn(2, 3, 32)
  targets = torch.tensor([[4, 5, 6], [7, -100, -100]])
  valid_logits = torch.cat((logits[0], logits[1, :1]))
  valid_targets = torch.tensor([4, 5, 6, 7])
  expected = torch.nn.functional.cross_entropy(
    valid_logits,
    valid_targets,
    reduction="sum",
  ) + 0.5 * torch.nn.functional.cross_entropy(
    valid_logits[:-1],
    valid_targets[1:],
    reduction="sum",
  )

  loss = model._multi_token_loss(logits, targets)

  torch.testing.assert_close(loss, expected / 4)


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


def test_nanogpt_speedrun_model_updates_runtime_stage_features() -> None:
  model = NanoGPTSpeedrunModel(_config())

  model.set_mtp_weights((1.0, 0.25))
  model.set_attention_windows(short=2, long=4)
  before_short_frequency = model.blocks[0].attn.yarn.angular_freq.clone()
  before_frequency = model.blocks[1].attn.yarn.angular_freq.clone()
  model.set_attention_windows(short=4, long=8)
  logits, _ = model(torch.randint(0, 32, (2, 8)))

  assert model.active_mtp_weights == (1.0, 0.25)
  assert model.active_short_window == 4
  assert model.active_long_window == 8
  assert not torch.equal(model.blocks[1].attn.yarn.angular_freq, before_frequency)
  assert not torch.equal(
    model.blocks[0].attn.yarn.angular_freq,
    before_short_frequency,
  )
  assert logits is not None


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
  assert model.blocks[0].attn.paired_heads is True
  assert model.blocks[1].attn.paired_heads is False


def test_nanogpt_speedrun_model_initializes_and_runs_mudd_topology() -> None:
  config = NanoGPTSpeedrunConfig(
    vocab_size=32,
    max_seq_len=8,
    d_model=16,
    n_layers=11,
    n_heads=2,
    d_ff=32,
    mudd_hidden_dim=8,
    value_residual=False,
    logit_sigmoid_scale=5.0,
  )
  model = NanoGPTSpeedrunModel(config)
  assert model.mudd is not None
  x = torch.zeros(1, 8, 16)

  pre = model.mudd(x, route=0, num_coefficients=14)
  post = model.mudd(x, route=1, num_coefficients=5)
  logits, _ = model(torch.randint(0, 32, (1, 8)))

  torch.testing.assert_close(pre[6], 2 * torch.ones_like(pre[6]))
  torch.testing.assert_close(
    pre[8],
    config.residual_decay**0.5 * torch.ones_like(pre[8]),
  )
  torch.testing.assert_close(post[1], -0.5 * torch.ones_like(post[1]))
  assert logits is not None
  assert torch.isfinite(logits).all()


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
    "paired_head_layers": (0,),
    "long_window_layers": (1,),
    "shared_attention_source_layer": None,
    "shared_attention_start_layer": None,
    "value_embedding_layers": (1,),
    "mudd": False,
  }
  values.update(overrides)
  return NanoGPTSpeedrunConfig(**values)
