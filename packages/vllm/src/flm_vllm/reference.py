"""Native vLLM implementation of the FLM reference model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from flm_llm import ReferenceModel, ReferenceModelConfig
from torch import nn

try:  # pragma: no cover - exercised only with the optional vLLM runtime.
  from vllm.distributed import get_tensor_model_parallel_world_size
  from vllm.model_executor.layers.activation import SiluAndMul
  from vllm.model_executor.layers.attention import Attention
  from vllm.model_executor.layers.layernorm import RMSNorm
  from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
  )
  from vllm.model_executor.layers.logits_processor import LogitsProcessor
  from vllm.model_executor.layers.rotary_embedding import get_rope
  from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
  )
  from vllm.model_executor.model_loader.weight_utils import default_weight_loader
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
  if exc.name != "vllm":
    raise
  _VLLM_AVAILABLE = False
else:
  _VLLM_AVAILABLE = True


if _VLLM_AVAILABLE:  # pragma: no branch - definitions depend on optional vLLM.

  class FlmAttention(nn.Module):
    """FLM self-attention expressed with vLLM cache-aware primitives."""

    def __init__(self, *, config, cache_config, quant_config, prefix: str) -> None:
      super().__init__()
      hidden_size = int(config.hidden_size)
      total_heads = int(config.num_attention_heads)
      if hidden_size % total_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
      tp_size = get_tensor_model_parallel_world_size()
      if total_heads % tp_size:
        raise ValueError(
          "num_attention_heads must be divisible by tensor parallel size"
        )

      self.num_heads = total_heads // tp_size
      self.head_dim = hidden_size // total_heads
      self.qkv = QKVParallelLinear(
        hidden_size=hidden_size,
        head_size=self.head_dim,
        total_num_heads=total_heads,
        total_num_kv_heads=total_heads,
        bias=bool(getattr(config, "attention_bias", False)),
        quant_config=quant_config,
        prefix=f"{prefix}.qkv",
      )
      self.out = RowParallelLinear(
        input_size=hidden_size,
        output_size=hidden_size,
        bias=bool(getattr(config, "attention_bias", False)),
        quant_config=quant_config,
        prefix=f"{prefix}.out",
      )
      self.rope = get_rope(
        self.head_dim,
        max_position=int(config.max_position_embeddings),
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
        is_neox_style=True,
      )
      self.attention = Attention(
        self.num_heads,
        self.head_dim,
        self.head_dim**-0.5,
        num_kv_heads=self.num_heads,
        cache_config=cache_config,
        quant_config=quant_config,
        prefix=f"{prefix}.attention",
      )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
      qkv, _ = self.qkv(x)
      q, k, v = qkv.chunk(3, dim=-1)
      q, k = self.rope(positions, q, k)
      output = self.attention(q, k, v)
      output, _ = self.out(output)
      return output

  class FlmSwiGLU(nn.Module):
    def __init__(self, *, config, quant_config, prefix: str) -> None:
      super().__init__()
      bias = bool(getattr(config, "mlp_bias", False))
      self.up = MergedColumnParallelLinear(
        input_size=int(config.hidden_size),
        output_sizes=[int(config.intermediate_size)] * 2,
        bias=bias,
        quant_config=quant_config,
        prefix=f"{prefix}.up",
      )
      self.down = RowParallelLinear(
        input_size=int(config.intermediate_size),
        output_size=int(config.hidden_size),
        bias=bias,
        quant_config=quant_config,
        prefix=f"{prefix}.down",
      )
      self.activation = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      x, _ = self.up(x)
      x = self.activation(x)
      x, _ = self.down(x)
      return x

  class FlmBlock(nn.Module):
    def __init__(self, *, vllm_config, prefix: str) -> None:
      super().__init__()
      config = vllm_config.model_config.hf_config
      self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
      self.attn = FlmAttention(
        config=config,
        cache_config=vllm_config.cache_config,
        quant_config=vllm_config.quant_config,
        prefix=f"{prefix}.attn",
      )
      self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
      self.ffn = FlmSwiGLU(
        config=config,
        quant_config=vllm_config.quant_config,
        prefix=f"{prefix}.ffn",
      )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
      x = x + self.attn(positions, self.attn_norm(x))
      return x + self.ffn(self.ffn_norm(x))

  class FlmReferenceForCausalLM(nn.Module):
    """The FLM reference topology implemented directly for vLLM.

    Logical model dimensions are never changed to accommodate a backend. If a
    vLLM backend does not support an FLM head size, engine construction fails
    instead of running a padded, backend-specific approximation.
    """

    packed_modules_mapping = {
      "qkv": ["q", "k", "v"],
      "up": ["gate", "value"],
    }

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
      super().__init__()
      del prefix
      self.config = vllm_config.model_config.hf_config
      quant_config = vllm_config.quant_config
      self.token_embedding = VocabParallelEmbedding(
        self.config.vocab_size,
        self.config.hidden_size,
        quant_config=quant_config,
      )
      self.blocks = nn.ModuleList(
        FlmBlock(vllm_config=vllm_config, prefix=f"blocks.{index}")
        for index in range(self.config.num_hidden_layers)
      )
      self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
      head = ParallelLMHead(
        self.config.vocab_size,
        self.config.hidden_size,
        quant_config=quant_config,
        prefix="lm_head",
      )
      self.lm_head = head.tie_weights(self.token_embedding)
      self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
      return self.token_embedding(input_ids)

    def forward(
      self,
      input_ids: torch.Tensor | None,
      positions: torch.Tensor,
      intermediate_tensors=None,
      inputs_embeds: torch.Tensor | None = None,
      **_: object,
    ) -> torch.Tensor:
      if intermediate_tensors is not None:
        raise ValueError("pipeline parallelism is not supported by the FLM adapter")
      if inputs_embeds is None:
        if input_ids is None:
          raise ValueError("input_ids or inputs_embeds is required")
        x = self.token_embedding(input_ids)
      else:
        x = inputs_embeds
      for block in self.blocks:
        x = block(positions, x)
      return self.norm(x)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
      return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
      params = dict(self.named_parameters())
      loaded = set()
      for raw_name, tensor in weights:
        name = raw_name.removeprefix("model.")
        if name == "lm_head.weight":
          continue
        param = params.get(name)
        if param is None:
          raise ValueError(f"unsupported FLM checkpoint tensor: {raw_name}")
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, tensor)
        loaded.add(name)
      return loaded


else:

  class FlmReferenceForCausalLM(nn.Module):
    """Native FLM fallback used by export tests when vLLM is unavailable."""

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
      input_ids: torch.Tensor | None,
      positions: torch.Tensor | None = None,
      intermediate_tensors=None,
      inputs_embeds: torch.Tensor | None = None,
      **_: object,
    ) -> torch.Tensor:
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
      return self.model.token_embedding(input_ids)

    def compute_logits(
      self,
      hidden_states: torch.Tensor,
      sampling_metadata=None,
    ) -> torch.Tensor:
      del sampling_metadata
      return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
      state = {name.removeprefix("model."): tensor for name, tensor in weights}
      self.model.load_state_dict(state)
      return {f"model.{name}" for name in state}
