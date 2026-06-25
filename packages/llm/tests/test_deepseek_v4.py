import torch
from flm_llm import DeepSeekV4, DeepSeekV4Config
from flm_modules import (
  DeepSeekMLA,
  DeepSeekMoE,
  DeepSeekV4HyperConnection,
  DeepSeekV4HyperHead,
  SwiGLU,
)


def test_deepseek_v4_returns_logits_and_loss() -> None:
  model = DeepSeekV4(_tiny_config())
  input_ids = torch.randint(0, 32, (2, 8))
  targets = torch.randint(0, 32, (2, 8))

  logits, loss = model(input_ids, targets)

  assert logits.shape == (2, 8, 32)
  assert loss is not None
  assert loss.item() > 0


def test_deepseek_v4_uses_mla_and_moe_blocks() -> None:
  model = DeepSeekV4(_tiny_config(n_layers=3, dense_layers=1))

  assert isinstance(model.hc_head, DeepSeekV4HyperHead)
  assert isinstance(model.blocks[0].attn_hc, DeepSeekV4HyperConnection)
  assert isinstance(model.blocks[0].ffn_hc, DeepSeekV4HyperConnection)
  assert isinstance(model.blocks[0].attn, DeepSeekMLA)
  assert isinstance(model.blocks[0].ffn, SwiGLU)
  assert isinstance(model.blocks[1].ffn, DeepSeekMoE)
  assert isinstance(model.blocks[2].ffn, DeepSeekMoE)
  assert model.blocks[1].ffn.expert_kind == "v4"


def test_deepseek_v4_backpropagates() -> None:
  model = DeepSeekV4(_tiny_config())
  input_ids = torch.randint(0, 32, (2, 8))
  targets = torch.randint(0, 32, (2, 8))

  _, loss = model(input_ids, targets)
  assert loss is not None
  loss.backward()

  assert model.token_embedding.weight.grad is not None
  assert model.hc_head.hc_fn.grad is not None
  assert model.blocks[0].attn_hc.fn.grad is not None
  assert model.blocks[0].ffn_hc.fn.grad is not None
  assert model.blocks[0].attn.kv_a_proj_with_mqa.weight.grad is not None
  assert model.blocks[-1].ffn.gate.weight.grad is not None


def _tiny_config(
  n_layers: int = 2,
  dense_layers: int = 1,
) -> DeepSeekV4Config:
  return DeepSeekV4Config(
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
    moe_d_ff=16,
    n_routed_experts=4,
    n_shared_experts=1,
    n_experts_per_token=2,
    n_group=2,
    topk_group=1,
    dense_layers=dense_layers,
  )
