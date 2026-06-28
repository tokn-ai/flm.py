import torch
from flm_llm import DSTiny, DSTinyConfig
from flm_modules import DeepSeekMLA, RopeLayout, SwiGLU


def test_ds_tiny_returns_logits_and_loss() -> None:
  model = DSTiny(_tiny_config())
  input_ids = torch.randint(0, 32, (2, 8))
  targets = torch.randint(0, 32, (2, 8))

  logits, loss = model(input_ids, targets)

  assert logits.shape == (2, 8, 32)
  assert loss is not None
  assert loss.item() > 0


def test_ds_tiny_uses_mla_and_dense_blocks() -> None:
  model = DSTiny(_tiny_config(n_layers=3))

  assert len(model.blocks) == 3
  assert all(isinstance(block.attn, DeepSeekMLA) for block in model.blocks)
  assert all(
    block.attn.rope.layout == RopeLayout.DEEPSEEK_V32 for block in model.blocks
  )
  assert all(isinstance(block.ffn, SwiGLU) for block in model.blocks)


def test_ds_tiny_backpropagates() -> None:
  model = DSTiny(_tiny_config())
  input_ids = torch.randint(0, 32, (2, 8))
  targets = torch.randint(0, 32, (2, 8))

  _, loss = model(input_ids, targets)
  assert loss is not None
  loss.backward()

  assert model.token_embedding.weight.grad is not None
  assert model.blocks[0].attn.kv_a_proj_with_mqa.weight.grad is not None
  assert model.blocks[0].ffn.up.weight.grad is not None


def _tiny_config(n_layers: int = 2) -> DSTinyConfig:
  return DSTinyConfig(
    vocab_size=32,
    max_seq_len=8,
    d_model=16,
    n_layers=n_layers,
    n_heads=2,
    q_lora_rank=8,
    kv_lora_rank=8,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=8,
    d_ff=16,
  )
