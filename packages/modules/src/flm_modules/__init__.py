"""Reusable neural network building blocks."""

from flm_modules.attentions import (
  AttentionBackend,
  DeepSeekDSA,
  DeepSeekDSAIndexer,
  DeepSeekMLA,
  DeepSeekV4Attention,
  DeepSeekV4AttentionKind,
  DeepSeekV4CSACompressor,
  DeepSeekV4HCACompressor,
  DeepSeekV4Indexer,
  DeepSeekV4IndexerScorer,
  QKNormSelfAttention,
  SelfAttention,
  scaled_dot_product_attention,
)
from flm_modules.feed_forward import ReLUSquared, SwiGLU
from flm_modules.hyper import (
  DeepSeekV4HyperConnection,
  DeepSeekV4HyperHead,
  UnweightedRMSNorm,
)
from flm_modules.kernels.tilelang import (
  tilelang_flash_attention,
  tilelang_linear_cross_entropy,
)
from flm_modules.linear import GroupedLinear
from flm_modules.losses import LossBackend, language_model_loss, linear_cross_entropy
from flm_modules.moe import (
  DeepSeekMoE,
  DeepSeekTopKRouter,
  DeepSeekV4MLP,
  ExpertKind,
  RouterScoring,
)
from flm_modules.norm import RMSNorm
from flm_modules.optim import CompositeOptimizer, Muon, configure_adamw, configure_muon
from flm_modules.rope import RopeLayout, RotaryEmbedding, apply_rotary

__all__ = [
  "AttentionBackend",
  "CompositeOptimizer",
  "DeepSeekDSA",
  "DeepSeekDSAIndexer",
  "DeepSeekMLA",
  "DeepSeekMoE",
  "DeepSeekTopKRouter",
  "DeepSeekV4Attention",
  "DeepSeekV4AttentionKind",
  "DeepSeekV4CSACompressor",
  "DeepSeekV4HCACompressor",
  "DeepSeekV4Indexer",
  "DeepSeekV4IndexerScorer",
  "DeepSeekV4HyperConnection",
  "DeepSeekV4HyperHead",
  "DeepSeekV4MLP",
  "ExpertKind",
  "GroupedLinear",
  "language_model_loss",
  "linear_cross_entropy",
  "LossBackend",
  "Muon",
  "RMSNorm",
  "ReLUSquared",
  "RouterScoring",
  "RopeLayout",
  "RotaryEmbedding",
  "SelfAttention",
  "QKNormSelfAttention",
  "SwiGLU",
  "UnweightedRMSNorm",
  "apply_rotary",
  "configure_adamw",
  "configure_muon",
  "scaled_dot_product_attention",
  "tilelang_flash_attention",
  "tilelang_linear_cross_entropy",
]
