"""Model builders."""

from __future__ import annotations

import torch
from flm_llm import (
  DeepSeekV4,
  DeepSeekV4Config,
  DSTiny,
  DSTinyConfig,
  ReferenceModel,
  ReferenceModelConfig,
)

from flm_train.types import TrainConfig


def build_model(config: TrainConfig, vocab_size: int) -> torch.nn.Module:
  if config.model_name == "reference":
    return ReferenceModel(
      ReferenceModelConfig(
        vocab_size=vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
      )
    )
  if config.model_name == "deepseek_v4":
    return DeepSeekV4(
      DeepSeekV4Config(
        vocab_size=vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        rope_head_dim=config.rope_head_dim,
        o_lora_rank=config.o_lora_rank,
        o_groups=config.o_groups,
        attention_layer_types=config.attention_layer_types,
        compress_rate_csa=config.compress_rate_csa,
        compress_rate_hca=config.compress_rate_hca,
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        moe_d_ff=config.d_ff,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_experts_per_token=config.n_experts_per_token,
        n_group=config.n_group,
        topk_group=config.topk_group,
        dense_layers=config.dense_layers,
      )
    )
  if config.model_name == "ds_tiny":
    return DSTiny(
      DSTinyConfig(
        vocab_size=vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        d_ff=config.d_ff,
      )
    )
  raise ValueError(f"unknown model_name: {config.model_name}")
