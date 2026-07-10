"""Reference model adapter for vLLM."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from flm_llm import ReferenceModel, ReferenceModelConfig
from torch import nn

try:  # pragma: no cover - exercised in the optional vLLM runtime.
  from vllm.model_executor.models.llama import LlamaForCausalLM as _VllmLlama
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
  if exc.name != "vllm":
    raise
  _VllmLlama = nn.Module
  _VLLM_AVAILABLE = False
else:
  _VLLM_AVAILABLE = True


class FlmReferenceForCausalLM(_VllmLlama):
  """Run FLM reference checkpoints with vLLM's native Llama engine.

  The FLM reference topology matches the dense Llama execution topology. The
  production adapter therefore reuses vLLM's attention, RoPE, and KV-cache
  implementation and translates only checkpoint names. Without the optional
  vLLM dependency, a native FLM fallback keeps export validation lightweight.
  """

  def __init__(self, *, vllm_config, prefix: str = "") -> None:
    if _VLLM_AVAILABLE:
      config = vllm_config.model_config.hf_config
      self._logical_head_dim = int(config.hidden_size) // int(
        config.num_attention_heads
      )
      self._physical_head_dim = _cpu_supported_head_dim(self._logical_head_dim)
      config.head_dim = self._physical_head_dim
      super().__init__(vllm_config=vllm_config, prefix=prefix)
      if self._physical_head_dim != self._logical_head_dim:
        scale = self._logical_head_dim**-0.5
        for layer in self.model.layers:
          layer.self_attn.scaling = scale
          layer.self_attn.attn.impl.scale = scale
      return

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
    input_ids: torch.Tensor | None,
    positions: torch.Tensor | None = None,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
    **_: object,
  ) -> torch.Tensor:
    if _VLLM_AVAILABLE:
      if positions is None:
        raise ValueError("positions are required by the vLLM runtime")
      return super().forward(
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
      )

    del positions, intermediate_tensors, inputs_embeds
    if input_ids is None:
      raise ValueError("input_ids are required by the native FLM fallback")
    if input_ids.ndim == 1:
      input_ids = input_ids.unsqueeze(0)
    logits, _ = self.model(input_ids)
    if logits is None:
      raise RuntimeError("reference model did not return logits")
    return logits.reshape(-1, logits.shape[-1])

  def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    if _VLLM_AVAILABLE:
      return super().embed_input_ids(input_ids)
    return self.model.token_embedding(input_ids)

  def compute_logits(
    self,
    hidden_states: torch.Tensor,
    sampling_metadata=None,
  ) -> torch.Tensor:
    del sampling_metadata
    if _VLLM_AVAILABLE:
      return super().compute_logits(hidden_states)
    return hidden_states

  def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    if _VLLM_AVAILABLE:
      mapped = (self._map_vllm_weight(name, tensor) for name, tensor in weights)
      return super().load_weights(mapped)

    loaded_names = set()
    state = {}
    for name, tensor in weights:
      loaded_names.add(name if name.startswith("model.") else f"model.{name}")
      state[name.removeprefix("model.")] = tensor
    missing, unexpected = self.model.load_state_dict(state, strict=False)
    if unexpected:
      raise RuntimeError(f"unexpected FLM checkpoint tensors: {sorted(unexpected)}")
    missing_names = set(missing) | {f"model.{name}" for name in missing}
    return loaded_names - missing_names

  def _map_vllm_weight(
    self,
    name: str,
    tensor: torch.Tensor,
  ) -> tuple[str, torch.Tensor]:
    mapped_name = _vllm_weight_name(name)
    if self._physical_head_dim == self._logical_head_dim:
      return mapped_name, tensor
    if mapped_name.endswith("self_attn.qkv_proj.weight") or mapped_name.endswith(
      "self_attn.qkv_proj.bias"
    ):
      tensor = _pad_qkv_heads(
        tensor,
        n_heads=int(self.config.num_attention_heads),
        logical_head_dim=self._logical_head_dim,
        physical_head_dim=self._physical_head_dim,
      )
    elif mapped_name.endswith("self_attn.o_proj.weight"):
      tensor = _pad_attention_output(
        tensor,
        n_heads=int(self.config.num_attention_heads),
        logical_head_dim=self._logical_head_dim,
        physical_head_dim=self._physical_head_dim,
      )
    return mapped_name, tensor


_BLOCK_WEIGHT_NAMES = {
  "attn_norm": "input_layernorm",
  "attn.qkv": "self_attn.qkv_proj",
  "attn.out": "self_attn.o_proj",
  "ffn_norm": "post_attention_layernorm",
  "ffn.up": "mlp.gate_up_proj",
  "ffn.down": "mlp.down_proj",
}


def _vllm_weight_name(name: str) -> str:
  name = name.removeprefix("model.")
  if name.startswith("token_embedding."):
    return "model.embed_tokens." + name.removeprefix("token_embedding.")
  if name.startswith("norm.") or name.startswith("lm_head."):
    return name if name.startswith("lm_head.") else f"model.{name}"

  match = re.fullmatch(r"blocks\.(\d+)\.(.+)\.(weight|bias)", name)
  if match is None:
    raise ValueError(f"unsupported FLM checkpoint tensor: {name}")
  layer, component, parameter = match.groups()
  mapped_component = _BLOCK_WEIGHT_NAMES.get(component)
  if mapped_component is None:
    raise ValueError(f"unsupported FLM checkpoint tensor: {name}")
  return f"model.layers.{layer}.{mapped_component}.{parameter}"


def _cpu_supported_head_dim(logical_head_dim: int) -> int:
  from vllm.platforms import current_platform

  if current_platform.device_type != "cpu":
    return logical_head_dim
  from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

  supported = CPUAttentionBackend.get_supported_head_sizes()
  for head_dim in supported:
    if head_dim >= logical_head_dim and head_dim % logical_head_dim == 0:
      return head_dim
  raise ValueError(
    f"FLM head dimension {logical_head_dim} exceeds vLLM CPU support: {supported}"
  )


def _pad_qkv_heads(
  tensor: torch.Tensor,
  *,
  n_heads: int,
  logical_head_dim: int,
  physical_head_dim: int,
) -> torch.Tensor:
  tail = tensor.shape[1:]
  source = tensor.reshape(3, n_heads, logical_head_dim, *tail)
  padded = tensor.new_zeros((3, n_heads, physical_head_dim, *tail))
  half = logical_head_dim // 2
  if physical_head_dim % logical_head_dim != 0:
    raise ValueError(
      "physical head padding must be an integer multiple of logical head size"
    )
  stride = physical_head_dim // logical_head_dim
  rope_indices = torch.arange(half, device=tensor.device) * stride
  padded[:2, :, rope_indices] = source[:2, :, :half]
  padded[:2, :, physical_head_dim // 2 + rope_indices] = source[:2, :, half:]
  padded[2, :, :logical_head_dim] = source[2]
  return padded.reshape(3 * n_heads * physical_head_dim, *tail)


def _pad_attention_output(
  tensor: torch.Tensor,
  *,
  n_heads: int,
  logical_head_dim: int,
  physical_head_dim: int,
) -> torch.Tensor:
  source = tensor.reshape(tensor.shape[0], n_heads, logical_head_dim)
  padded = tensor.new_zeros((tensor.shape[0], n_heads, physical_head_dim))
  padded[:, :, :logical_head_dim] = source
  return padded.flatten(1)
