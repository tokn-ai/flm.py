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
  model_config = config.model
  if model_config.kind == "reference":
    return ReferenceModel(
      ReferenceModelConfig(
        vocab_size=vocab_size,
        max_seq_len=config.data.seq_len,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
      )
    )
  if model_config.kind == "deepseek_v4":
    return DeepSeekV4(
      DeepSeekV4Config(
        vocab_size=vocab_size,
        max_seq_len=config.data.seq_len,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        head_dim=model_config.head_dim,
        q_lora_rank=model_config.q_lora_rank,
        kv_lora_rank=model_config.kv_lora_rank,
        qk_nope_head_dim=model_config.qk_nope_head_dim,
        qk_rope_head_dim=model_config.qk_rope_head_dim,
        v_head_dim=model_config.v_head_dim,
        rope_head_dim=model_config.rope_head_dim,
        o_lora_rank=model_config.o_lora_rank,
        o_groups=model_config.o_groups,
        attention_layer_types=model_config.attention_layer_types,
        compress_rate_csa=model_config.compress_rate_csa,
        compress_rate_hca=model_config.compress_rate_hca,
        index_n_heads=model_config.index_n_heads,
        index_head_dim=model_config.index_head_dim,
        index_topk=model_config.index_topk,
        moe_d_ff=model_config.d_ff,
        n_routed_experts=model_config.n_routed_experts,
        n_shared_experts=model_config.n_shared_experts,
        n_experts_per_token=model_config.n_experts_per_token,
        n_group=model_config.n_group,
        topk_group=model_config.topk_group,
        dense_layers=model_config.dense_layers,
      )
    )
  if model_config.kind == "ds_tiny":
    return DSTiny(
      DSTinyConfig(
        vocab_size=vocab_size,
        max_seq_len=config.data.seq_len,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        q_lora_rank=model_config.q_lora_rank,
        kv_lora_rank=model_config.kv_lora_rank,
        qk_nope_head_dim=model_config.qk_nope_head_dim,
        qk_rope_head_dim=model_config.qk_rope_head_dim,
        v_head_dim=model_config.v_head_dim,
        d_ff=model_config.d_ff,
      )
    )
  raise ValueError(f"unknown model kind: {model_config.kind}")
