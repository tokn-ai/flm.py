"""Eager reference model for the nanoGPT short-track speedrun architecture."""

from __future__ import annotations

import torch
from flm_modules import (
  BigramHashEmbedding,
  MultiwayDynamicDenseConnections,
  QKNormSelfAttention,
  ReLUSquared,
  RMSNorm,
  TokenSmear,
)
from flm_modules.losses import language_model_loss
from torch import nn
from torch.nn import functional as F

from flm_llm.config import NanoGPTSpeedrunConfig


class NanoGPTSpeedrunBlock(nn.Module):
  def __init__(self, config: NanoGPTSpeedrunConfig, *, layer_index: int) -> None:
    super().__init__()
    self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.attn = QKNormSelfAttention(
      d_model=config.d_model,
      n_heads=config.n_heads,
      bias=config.bias,
      rope_base=config.rope_base,
      norm_eps=config.norm_eps,
      backend=config.attention_backend,
      zero_init_out=True,
      paired_heads=layer_index in config.paired_head_layers,
      speedrun_yarn=True,
      max_seq_len=config.max_seq_len,
    )
    self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.ffn = ReLUSquared(
      d_model=config.d_model,
      d_ff=config.d_ff,
      bias=config.bias,
      zero_init_down=True,
    )

  def forward(
    self,
    x: torch.Tensor,
    *,
    value_residual: torch.Tensor | None = None,
    value_mix: torch.Tensor | float | None = None,
    token_value_embedding: torch.Tensor | None = None,
    value_gate_weight: torch.Tensor | None = None,
    additional_auxiliary_values: torch.Tensor | None = None,
    partial_key_offset: bool = False,
    output_gate_weight: torch.Tensor | None = None,
    xsa_alpha: torch.Tensor | None = None,
    attention_window: int | None = None,
    skip_attention: bool = False,
    residual_scales: torch.Tensor | None = None,
    post_scales: torch.Tensor | None = None,
    residual_injection: torch.Tensor | None = None,
    attention_input: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if skip_attention:
      attn_output = torch.zeros_like(x)
      values = (
        value_residual
        if value_residual is not None
        else x.new_zeros(
          x.shape[0],
          self.attn.n_heads,
          x.shape[1],
          self.attn.head_dim,
        )
      )
    else:
      attn_input = self.attn_norm(x if attention_input is None else attention_input)
      auxiliary_values = self._value_embedding_residual(
        attn_input,
        token_value_embedding=token_value_embedding,
        value_gate_weight=value_gate_weight,
      )
      if additional_auxiliary_values is not None:
        auxiliary_values = (
          additional_auxiliary_values
          if auxiliary_values is None
          else auxiliary_values + additional_auxiliary_values
        )
      attn_output, values = self.attn(
        attn_input,
        value_residual=value_residual,
        value_mix=value_mix,
        auxiliary_values=auxiliary_values,
        partial_key_offset=partial_key_offset,
        output_gate_weight=output_gate_weight,
        xsa_alpha=xsa_alpha,
        attention_window=attention_window,
      )
    residual_scales = x.new_ones(2) if residual_scales is None else residual_scales
    post_scales = x.new_ones(2) if post_scales is None else post_scales
    x = residual_scales[0] * x + post_scales[0] * attn_output
    if residual_injection is not None:
      x = x + residual_injection
    x = residual_scales[1] * x + post_scales[1] * self.ffn(self.ffn_norm(x))
    return x, values

  def _value_embedding_residual(
    self,
    attn_input: torch.Tensor,
    *,
    token_value_embedding: torch.Tensor | None,
    value_gate_weight: torch.Tensor | None,
  ) -> torch.Tensor | None:
    if token_value_embedding is None:
      return None
    if value_gate_weight is None:
      raise ValueError("value_gate_weight is required for token value embeddings")
    if value_gate_weight.ndim != 2:
      raise ValueError("value_gate_weight must be a matrix")
    gate_input_dim = value_gate_weight.shape[-1]
    if gate_input_dim % 2:
      raise ValueError("value gate input dimension must be even")
    feature_dim = gate_input_dim // 2
    gate_input = torch.cat(
      (
        attn_input[..., :feature_dim],
        token_value_embedding[..., :feature_dim],
      ),
      dim=-1,
    )
    batch_size, seq_len, _ = attn_input.shape
    gate = 2 * torch.sigmoid(F.linear(gate_input, value_gate_weight))
    values = token_value_embedding.view(
      batch_size,
      seq_len,
      self.attn.n_heads,
      self.attn.head_dim,
    )
    return (gate.unsqueeze(-1) * values).transpose(1, 2)


class NanoGPTSpeedrunModel(nn.Module):
  """Portable semantic baseline for current nanoGPT speedrun features.

  This intentionally uses ordinary PyTorch operations. Distributed optimizer
  sharding, FP8, FlashAttention 3, and fused H100 kernels belong to a later
  execution backend and do not change this model's public contract.
  """

  def __init__(self, config: NanoGPTSpeedrunConfig) -> None:
    super().__init__()
    if config.n_layers < 1:
      raise ValueError("n_layers must be positive")
    if config.logit_softcap is not None and config.logit_softcap <= 0:
      raise ValueError("logit_softcap must be positive")
    if config.logit_sigmoid_scale is not None:
      if config.logit_sigmoid_scale <= 0:
        raise ValueError("logit_sigmoid_scale must be positive")
      if config.logit_sigmoid_temperature <= 0:
        raise ValueError("logit_sigmoid_temperature must be positive")
    if not config.mtp_weights or config.mtp_weights[0] <= 0:
      raise ValueError("mtp_weights must start with a positive primary weight")
    if any(weight < 0 for weight in config.mtp_weights):
      raise ValueError("mtp_weights must be non-negative")
    if not 1 <= config.attention_gate_dim <= config.d_model:
      raise ValueError("attention_gate_dim must be in [1, d_model]")
    if config.attention_free_layer == 0:
      raise ValueError("the first layer cannot be attention-free")
    if (config.shared_attention_source_layer is None) != (
      config.shared_attention_start_layer is None
    ):
      raise ValueError("shared attention source/start must both be set")
    if config.shared_attention_source_layer is not None:
      if not (
        0
        <= config.shared_attention_source_layer
        < config.shared_attention_start_layer
        < config.n_layers
      ):
        raise ValueError("shared attention source/start layers are invalid")
    if config.residual_decay <= 0:
      raise ValueError("residual_decay must be positive")
    if config.n_heads % 2 and config.paired_head_layers:
      raise ValueError("paired_head_layers require an even number of heads")
    invalid_paired_layers = set(config.paired_head_layers) - set(range(config.n_layers))
    if invalid_paired_layers:
      raise ValueError("paired_head_layers contains an invalid layer index")
    invalid_long_window_layers = set(config.long_window_layers) - set(
      range(config.n_layers)
    )
    if invalid_long_window_layers:
      raise ValueError("long_window_layers contains an invalid layer index")
    invalid_value_layers = set(config.value_embedding_layers) - set(
      range(config.n_layers)
    )
    if invalid_value_layers:
      raise ValueError("value_embedding_layers contains an invalid layer index")
    if len(set(config.value_embedding_layers)) != len(config.value_embedding_layers):
      raise ValueError("value_embedding_layers must be unique")
    if (
      config.value_embedding_gate_dim % 2
      or config.value_embedding_gate_dim > 2 * config.d_model
    ):
      raise ValueError("value_embedding_gate_dim must be even and at most 2*d_model")
    if config.mudd:
      if config.n_layers < 11:
        raise ValueError("MUDD speedrun topology requires at least 11 layers")
      if 1 not in config.value_embedding_layers:
        raise ValueError("MUDD speedrun topology requires a layer-1 value embedding")
    if (config.block_skip_from is None) != (config.block_skip_to is None):
      raise ValueError("block skip endpoints must both be set or both be None")
    if config.block_skip_from is not None:
      if not 0 <= config.block_skip_from < config.block_skip_to < config.n_layers:
        raise ValueError("block skip endpoints must be ordered layer indices")

    self.config = config
    self.active_mtp_weights = config.mtp_weights
    self.active_short_window: int | None = None
    self.active_long_window: int | None = None
    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.005)
    self.token_smear = (
      TokenSmear(config.d_model, gate_dim=config.smear_gate_dim)
      if config.token_smear
      else None
    )
    if config.bigram_vocab_size is not None:
      if not 1 <= config.bigram_dim <= config.d_model:
        raise ValueError("bigram_dim must be in [1, d_model]")
      self.bigram_embedding = BigramHashEmbedding(
        config.bigram_vocab_size,
        config.bigram_dim,
        sign_table_rows=config.bigram_sign_table_rows,
      )
      self.bigram_injection_weights = nn.Parameter(torch.full((config.n_layers,), 0.05))
    else:
      self.bigram_embedding = None
      self.register_parameter("bigram_injection_weights", None)
    self.blocks = nn.ModuleList(
      NanoGPTSpeedrunBlock(config, layer_index=layer_index)
      for layer_index in range(config.n_layers)
    )
    self.attention_gate_weights = nn.Parameter(
      torch.zeros(config.n_layers, config.n_heads, config.attention_gate_dim)
    )
    if config.xsa:
      self.xsa_alphas = nn.Parameter(torch.zeros(config.n_layers, config.n_heads))
    else:
      self.register_parameter("xsa_alphas", None)
    value_embedding_count = len(config.value_embedding_layers)
    self.value_embeddings = nn.Parameter(
      0.01
      * torch.randn(
        value_embedding_count,
        config.vocab_size,
        config.d_model,
      )
    )
    self.value_gate_weights = nn.Parameter(
      torch.zeros(
        value_embedding_count,
        config.n_heads,
        config.value_embedding_gate_dim,
      )
    )
    self._value_embedding_index = {
      layer_index: bank_index
      for bank_index, layer_index in enumerate(config.value_embedding_layers)
    }
    if config.mudd:
      self.mudd = MultiwayDynamicDenseConnections(
        config.d_model,
        hidden_dim=config.mudd_hidden_dim,
        output_scale=config.mudd_scale,
      )
      self._initialize_mudd_biases()
    else:
      self.mudd = None
    residual_init = config.residual_decay**0.5
    self.residual_scales = nn.Parameter(torch.full((config.n_layers, 2), residual_init))
    self.post_scales = nn.Parameter(torch.ones(config.n_layers, 2))
    self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    with torch.no_grad():
      self.lm_head.weight.copy_(self.token_embedding.weight)
    self.embeddings_tied = config.tie_embeddings

    if config.embedding_skip:
      self.embedding_skip_weights = nn.Parameter(torch.zeros(config.n_layers))
    else:
      self.register_parameter("embedding_skip_weights", None)
    if config.value_residual and config.n_layers > 1:
      self.value_mix_logits = nn.Parameter(torch.zeros(config.n_layers - 1))
    else:
      self.register_parameter("value_mix_logits", None)
    if config.block_skip_from is not None:
      self.block_skip_logit = nn.Parameter(torch.tensor(-1.5))
      self.block_skip_gate = nn.Linear(config.attention_gate_dim, 1, bias=False)
      nn.init.zeros_(self.block_skip_gate.weight)
    else:
      self.register_parameter("block_skip_logit", None)
      self.block_skip_gate = None

  def set_mtp_weights(self, weights: tuple[float, ...]) -> None:
    if not weights or weights[0] <= 0 or any(weight < 0 for weight in weights):
      raise ValueError("mtp weights are invalid")
    self.active_mtp_weights = tuple(float(weight) for weight in weights)

  def set_attention_windows(self, *, short: int, long: int) -> None:
    if short < 1 or long < short:
      raise ValueError("attention windows must satisfy 1 <= short <= long")
    if self.active_short_window is not None and self.active_long_window is not None:
      for layer_index, block in enumerate(self.blocks):
        old_window = (
          self.active_long_window
          if layer_index in self.config.long_window_layers
          else self.active_short_window
        )
        new_window = long if layer_index in self.config.long_window_layers else short
        block.attn.update_yarn_window(old_window, new_window)
    self.active_short_window = short
    self.active_long_window = long

  def untie_embeddings(self) -> None:
    if not self.embeddings_tied:
      return
    with torch.no_grad():
      self.lm_head.weight.copy_(self.token_embedding.weight)
    self.embeddings_tied = False

  @property
  def classifier_weight(self) -> torch.Tensor:
    if self.embeddings_tied:
      return self.token_embedding.weight
    return self.lm_head.weight

  def _initialize_mudd_biases(self) -> None:
    if self.mudd is None:
      return
    inverse_scale = 1.0 / self.mudd.output_scale
    residual_init = self.config.residual_decay**0.5
    with torch.no_grad():
      self.mudd.bias[0, 6:8].fill_(2.0 * inverse_scale)
      self.mudd.bias[0, 8].fill_(residual_init * inverse_scale)
      self.mudd.bias[0, 9].fill_(inverse_scale)
      self.mudd.bias[0, 11].fill_(0.05 * inverse_scale)
      self.mudd.bias[0, 12].fill_(residual_init * inverse_scale)
      self.mudd.bias[0, 13].fill_(inverse_scale)
      self.mudd.bias[1, 1].fill_(-0.5 * inverse_scale)

  def forward(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    return_logits: bool = True,
  ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if input_ids.ndim != 2:
      raise ValueError("input_ids must have shape (batch, seq_len)")
    if input_ids.shape[1] > self.config.max_seq_len:
      raise ValueError("sequence length exceeds config.max_seq_len")

    embeddings = self.token_embedding(input_ids)
    x = self.token_smear(embeddings) if self.token_smear is not None else embeddings
    bigram_values = (
      self.bigram_embedding(input_ids) if self.bigram_embedding is not None else None
    )
    padded_bigram_values = (
      None
      if bigram_values is None
      else F.pad(
        bigram_values,
        (0, self.config.d_model - self.config.bigram_dim),
      )
    )
    first_values = None
    block_skip = None
    skip_gate_input = F.rms_norm(embeddings, (self.config.d_model,))
    cache = {0: x}
    for layer_index, block in enumerate(self.blocks):
      residual_injection = None
      if self.embedding_skip_weights is not None:
        residual_injection = self.embedding_skip_weights[layer_index] * embeddings
      if padded_bigram_values is not None and self.bigram_injection_weights is not None:
        bigram_injection = (
          self.bigram_injection_weights[layer_index] * padded_bigram_values
        )
        residual_injection = (
          bigram_injection
          if residual_injection is None
          else residual_injection + bigram_injection
        )
      if layer_index == self.config.block_skip_to:
        if (
          block_skip is None
          or self.block_skip_logit is None
          or self.block_skip_gate is None
        ):
          raise RuntimeError("block skip source was not captured")
        gate = (
          2
          * self.block_skip_logit.sigmoid()
          * torch.sigmoid(
            self.block_skip_gate(skip_gate_input[..., : self.config.attention_gate_dim])
          )
        )
        x = x + gate * block_skip

      value_mix = None
      if first_values is not None and self.value_mix_logits is not None:
        value_mix = self.value_mix_logits[layer_index - 1].sigmoid()
      value_bank_index = self._value_embedding_index.get(layer_index)
      token_value_embedding = (
        None
        if value_bank_index is None
        else self.value_embeddings[value_bank_index][input_ids]
      )
      additional_auxiliary_values = None
      residual_scales = self.residual_scales[layer_index]
      post_scales = self.post_scales[layer_index]
      if self.mudd is not None and layer_index == self.config.n_layers - 1:
        cache[9] = x
        coefficients = self.mudd(x, route=0, num_coefficients=14)
        value_mudd = (
          coefficients[0] * cache[0] + coefficients[1] * cache[7] + coefficients[2] * x
        )
        additional_auxiliary_values = value_mudd.view(
          *value_mudd.shape[:2],
          self.config.n_heads,
          self.config.d_model // self.config.n_heads,
        ).transpose(1, 2)
        x = (
          (1 + coefficients[5]) * x
          + coefficients[3] * cache[0]
          + coefficients[4] * cache[7]
        )
        if token_value_embedding is not None:
          dynamic_gate = torch.cat(
            (coefficients[6], coefficients[7]),
            dim=-1,
          ).repeat_interleave(self.config.n_heads // 2, dim=-1)
          dynamic_values = (
            dynamic_gate.unsqueeze(-1)
            * token_value_embedding.view(
              *token_value_embedding.shape[:2],
              self.config.n_heads,
              self.config.d_model // self.config.n_heads,
            )
          ).transpose(1, 2)
          additional_auxiliary_values = additional_auxiliary_values + dynamic_values
          token_value_embedding = None
          value_bank_index = None
        residual_scales = torch.stack(
          (coefficients[8], coefficients[12]),
          dim=0,
        )
        post_scales = torch.stack(
          (coefficients[9], coefficients[13]),
          dim=0,
        )
        residual_injection = coefficients[10] * cache[0]
        if padded_bigram_values is not None:
          residual_injection = (
            residual_injection + coefficients[11] * padded_bigram_values
          )
      x, values = block(
        x,
        value_residual=first_values,
        value_mix=value_mix,
        token_value_embedding=token_value_embedding,
        value_gate_weight=None
        if value_bank_index is None
        else self.value_gate_weights[value_bank_index],
        additional_auxiliary_values=additional_auxiliary_values,
        partial_key_offset=layer_index in self.config.partial_key_offset_layers,
        output_gate_weight=self.attention_gate_weights[layer_index],
        xsa_alpha=None
        if self.xsa_alphas is None or block.attn.paired_heads
        else self.xsa_alphas[layer_index],
        attention_window=(
          self.active_long_window
          if layer_index in self.config.long_window_layers
          else self.active_short_window
        ),
        skip_attention=layer_index == self.config.attention_free_layer,
        residual_scales=residual_scales,
        post_scales=post_scales,
        residual_injection=residual_injection,
        attention_input=(
          None
          if self.config.shared_attention_start_layer is None
          or layer_index < self.config.shared_attention_start_layer
          else cache[self.config.shared_attention_source_layer]
        ),
      )
      if first_values is None and self.config.value_residual:
        first_values = values
      if layer_index == self.config.block_skip_from:
        block_skip = x
      if layer_index in {3, 7, 9}:
        cache[layer_index] = x
      if layer_index == self.config.shared_attention_source_layer:
        cache[layer_index] = x

    if self.mudd is not None:
      coefficients = self.mudd(x, route=1, num_coefficients=5)
      value_bank_index = self._value_embedding_index[1]
      first_token_values = self.value_embeddings[value_bank_index][input_ids]
      x = (
        x
        + coefficients[0] * cache[0]
        + coefficients[1] * cache[7]
        + coefficients[2] * cache[9]
        + coefficients[3] * first_token_values
        + coefficients[4] * cache[3]
      )

    hidden_states = self.norm(x)
    needs_logits_for_loss = targets is not None and (
      self.config.logit_sigmoid_scale is not None
      or self.config.logit_softcap is not None
      or (self.training and len(self.active_mtp_weights) > 1)
    )
    logits = (
      self._logits(hidden_states) if return_logits or needs_logits_for_loss else None
    )
    loss = self._loss(hidden_states, logits, targets)
    return logits if return_logits else None, loss

  def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
    logits = self.config.logit_scale * F.linear(
      hidden_states,
      self.classifier_weight,
    )
    if self.config.logit_sigmoid_scale is not None:
      logits = self.config.logit_sigmoid_scale * torch.sigmoid(
        (logits + self.config.logit_sigmoid_bias)
        / self.config.logit_sigmoid_temperature
      )
    elif self.config.logit_softcap is not None:
      cap = self.config.logit_softcap
      logits = cap * torch.tanh(logits / cap)
    return logits

  def _loss(
    self,
    hidden_states: torch.Tensor,
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
  ) -> torch.Tensor | None:
    if targets is None:
      return None
    if (
      self.config.logit_sigmoid_scale is not None
      or self.config.logit_softcap is not None
      or (self.training and len(self.active_mtp_weights) > 1)
    ):
      if logits is None:
        raise RuntimeError("softcapped loss requires logits")
      return self._multi_token_loss(logits, targets)
    return language_model_loss(
      hidden_states=hidden_states,
      classifier_weight=self.classifier_weight,
      targets=targets,
      backend=self.config.loss_backend,
      chunk_size=self.config.loss_chunk_size,
    )

  def _multi_token_loss(
    self,
    logits: torch.Tensor,
    targets: torch.Tensor,
  ) -> torch.Tensor:
    weights = self.active_mtp_weights if self.training else (1.0,)
    token_count = targets.numel()
    total = logits.new_zeros((), dtype=torch.float32)
    for offset, weight in enumerate(weights):
      if weight == 0 or offset >= targets.shape[1]:
        continue
      offset_logits = logits[:, : targets.shape[1] - offset]
      offset_targets = targets[:, offset:]
      total = total + weight * F.cross_entropy(
        offset_logits.flatten(0, 1).float(),
        offset_targets.flatten(),
        reduction="sum",
      )
    return total / token_count
