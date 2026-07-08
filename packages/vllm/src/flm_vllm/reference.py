"""Reference model adapter for vLLM."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from flm_llm import ReferenceModel, ReferenceModelConfig
from torch import nn


class FlmReferenceForCausalLM(nn.Module):
  """A vLLM-registered wrapper around the FLM reference architecture.

  This adapter intentionally preserves FLM checkpoint names. The first export
  path is for correctness and batched vLLM rollout; deeper vLLM-native KV-cache
  integration can replace this wrapper without changing exported checkpoints.
  """

  def __init__(self, *, vllm_config, prefix: str = "") -> None:
    super().__init__()
    del prefix
    config = vllm_config.model_config.hf_config
    self.config = config
    self.model = ReferenceModel(
      ReferenceModelConfig(
        vocab_size=int(config.vocab_size),
        max_seq_len=int(config.max_position_embeddings),
        d_model=int(config.hidden_size),
        n_layers=int(config.num_hidden_layers),
        n_heads=int(config.num_attention_heads),
        d_ff=int(config.intermediate_size),
        bias=bool(getattr(config, "attention_bias", False)),
        rope_base=float(getattr(config, "rope_theta", 10_000.0)),
        norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
      )
    )

  def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor | None = None,
    **_: object,
  ) -> torch.Tensor:
    del positions
    if input_ids.ndim == 1:
      input_ids = input_ids.unsqueeze(0)
    logits, _ = self.model(input_ids)
    if logits is None:
      raise RuntimeError("reference model did not return logits")
    return logits.reshape(-1, logits.shape[-1])

  def compute_logits(
    self,
    hidden_states: torch.Tensor,
    sampling_metadata,
  ) -> torch.Tensor:
    del sampling_metadata
    return hidden_states

  def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    state = {name.removeprefix("model."): tensor for name, tensor in weights}
    missing, unexpected = self.model.load_state_dict(state, strict=False)
    if unexpected:
      raise RuntimeError(f"unexpected FLM checkpoint tensors: {sorted(unexpected)}")
    return set(state) - set(missing)
