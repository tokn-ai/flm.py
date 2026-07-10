"""vLLM integration helpers for FLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from flm_vllm.export import (
    FlmReferenceVllmConfig,
    export_reference_checkpoint,
    reference_vllm_config,
  )
  from flm_vllm.importing import ImportedReferenceModel, import_reference_export
  from flm_vllm.registration import register_flm_models

__all__ = [
  "FlmReferenceVllmConfig",
  "ImportedReferenceModel",
  "export_reference_checkpoint",
  "import_reference_export",
  "reference_vllm_config",
  "register_flm_models",
]


def __getattr__(name: str):
  if name in {
    "FlmReferenceVllmConfig",
    "export_reference_checkpoint",
    "reference_vllm_config",
  }:
    from flm_vllm import export

    return getattr(export, name)
  if name in {"ImportedReferenceModel", "import_reference_export"}:
    from flm_vllm import importing

    return getattr(importing, name)
  if name == "register_flm_models":
    from flm_vllm.registration import register_flm_models

    return register_flm_models
  raise AttributeError(name)
