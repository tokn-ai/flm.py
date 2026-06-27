import torch
from flm_llm import ReferenceModel, ReferenceModelConfig
from flm_modules import configure_adamw
from flm_rl import (
  GRPOBatch,
  GRPOTrainer,
  PPOBatch,
  PPOTrainer,
  compute_group_advantages,
  sequence_log_probs,
)


def _model() -> ReferenceModel:
  torch.manual_seed(0)
  return ReferenceModel(
    ReferenceModelConfig(
      vocab_size=16,
      max_seq_len=8,
      d_model=8,
      n_layers=1,
      n_heads=2,
      d_ff=16,
    )
  )


def test_sequence_log_probs_aligns_next_tokens() -> None:
  model = _model()
  input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

  log_probs = sequence_log_probs(model, input_ids)

  assert log_probs.shape == (2, 3)
  assert torch.isfinite(log_probs).all()


def test_ppo_trainer_updates_policy() -> None:
  model = _model()
  input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
  old_log_probs = sequence_log_probs(model, input_ids).detach()
  optimizer = configure_adamw(model, learning_rate=1e-3, weight_decay=0.0)
  trainer = PPOTrainer(model, optimizer)
  before = model.token_embedding.weight.detach().clone()

  metrics = trainer.step(
    PPOBatch(
      input_ids=input_ids,
      old_log_probs=old_log_probs,
      advantages=torch.tensor([[1.0, 0.5, 0.25], [-1.0, -0.5, -0.25]]),
      action_mask=torch.ones_like(old_log_probs),
    )
  )

  assert torch.isfinite(torch.tensor(metrics.loss))
  assert metrics.approx_kl >= 0.0
  assert not torch.equal(before, model.token_embedding.weight.detach())


def test_compute_group_advantages_normalizes_per_group() -> None:
  advantages = compute_group_advantages(
    rewards=torch.tensor([1.0, 3.0, 2.0, 4.0]),
    group_ids=torch.tensor([0, 0, 1, 1]),
  )

  assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, -1.0, 1.0]))


def test_grpo_trainer_updates_policy() -> None:
  model = _model()
  input_ids = torch.tensor(
    [
      [1, 2, 3, 4],
      [1, 2, 5, 6],
      [7, 8, 9, 1],
      [7, 8, 2, 3],
    ]
  )
  old_log_probs = sequence_log_probs(model, input_ids).detach()
  optimizer = configure_adamw(model, learning_rate=1e-3, weight_decay=0.0)
  trainer = GRPOTrainer(model, optimizer)
  before = model.token_embedding.weight.detach().clone()

  metrics = trainer.step(
    GRPOBatch(
      input_ids=input_ids,
      old_log_probs=old_log_probs,
      rewards=torch.tensor([1.0, 3.0, 2.0, 4.0]),
      group_ids=torch.tensor([0, 0, 1, 1]),
      action_mask=torch.ones_like(old_log_probs),
    )
  )

  assert torch.isfinite(torch.tensor(metrics.loss))
  assert metrics.reward_mean == 2.5
  assert not torch.equal(before, model.token_embedding.weight.detach())
