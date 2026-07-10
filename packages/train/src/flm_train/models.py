"""Model builders."""

from __future__ import annotations

import torch
from flm_llm import (
  DeepSeekV4,
  DeepSeekV4Config,
  DSTiny,
  DSTinyConfig,
  NanoGPTSpeedrunConfig,
  NanoGPTSpeedrunModel,
  ReferenceModel,
  ReferenceModelConfig,
)

from flm_train.types import TrainConfig


def build_model(config: TrainConfig, vocab_size: int) -> torch.nn.Module:
  model_config = config.model
  if model_config.kind == "nanogpt_speedrun":
    return NanoGPTSpeedrunModel(
      NanoGPTSpeedrunConfig(
        vocab_size=vocab_size,
        max_seq_len=config.data.seq_len,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
        attention_backend=model_config.attention_backend,
        loss_backend=model_config.loss_backend,
        loss_chunk_size=model_config.loss_chunk_size,
        logit_softcap=model_config.logit_softcap,
        logit_scale=model_config.logit_scale,
        logit_sigmoid_scale=model_config.logit_sigmoid_scale,
        logit_sigmoid_bias=model_config.logit_sigmoid_bias,
        logit_sigmoid_temperature=model_config.logit_sigmoid_temperature,
        token_smear=model_config.token_smear,
        smear_gate_dim=model_config.smear_gate_dim,
        partial_key_offset_layers=model_config.partial_key_offset_layers,
        attention_gate_dim=model_config.attention_gate_dim,
        xsa=model_config.xsa,
        attention_free_layer=model_config.attention_free_layer,
        paired_head_layers=model_config.paired_head_layers,
        long_window_layers=model_config.long_window_layers,
        value_embedding_layers=model_config.value_embedding_layers,
        value_embedding_gate_dim=model_config.value_embedding_gate_dim,
        mudd=model_config.mudd,
        mudd_hidden_dim=model_config.mudd_hidden_dim,
        mudd_scale=model_config.mudd_scale,
        bigram_vocab_size=model_config.bigram_vocab_size,
        bigram_dim=model_config.bigram_dim,
        bigram_sign_table_rows=model_config.bigram_sign_table_rows,
        mtp_weights=model_config.mtp_weights,
        embedding_skip=model_config.embedding_skip,
        value_residual=model_config.value_residual,
        block_skip_from=model_config.block_skip_from,
        block_skip_to=model_config.block_skip_to,
        residual_decay=model_config.residual_decay,
        tie_embeddings=model_config.tie_embeddings,
      )
    )
  if model_config.kind == "reference":
    return ReferenceModel(
      ReferenceModelConfig(
        vocab_size=vocab_size,
        max_seq_len=config.data.seq_len,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
        attention_backend=model_config.attention_backend,
        loss_backend=model_config.loss_backend,
        loss_chunk_size=model_config.loss_chunk_size,
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
        attention_backend=model_config.attention_backend,
        loss_backend=model_config.loss_backend,
        loss_chunk_size=model_config.loss_chunk_size,
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
        attention_backend=model_config.attention_backend,
        loss_backend=model_config.loss_backend,
        loss_chunk_size=model_config.loss_chunk_size,
      )
    )
  raise ValueError(f"unknown model kind: {model_config.kind}")
