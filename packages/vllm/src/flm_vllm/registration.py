"""vLLM model registration."""

from __future__ import annotations


def register_flm_models() -> None:
  """Register FLM out-of-tree models with vLLM."""
  try:
    from vllm import ModelRegistry
  except ImportError as exc:  # pragma: no cover - depends on optional vLLM.
    raise RuntimeError(
      "vLLM is not installed. Install vllm in the runtime environment."
    ) from exc

  from flm_vllm.reference import FlmReferenceForCausalLM

  ModelRegistry.register_model("FlmReferenceForCausalLM", FlmReferenceForCausalLM)
