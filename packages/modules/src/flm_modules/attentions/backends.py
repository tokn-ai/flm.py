"""Scaled dot-product attention backend dispatch."""

from __future__ import annotations

from enum import StrEnum

import torch
from torch.nn import functional as F


class AttentionBackend(StrEnum):
  TORCH = "torch"
  FLASH_ATTENTION2 = "flash_attention2"
  TILELANG = "tilelang"


def scaled_dot_product_attention(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  *,
  backend: AttentionBackend | str = AttentionBackend.TORCH,
  attn_mask: torch.Tensor | None = None,
  causal: bool = True,
  scale: float | None = None,
) -> torch.Tensor:
  backend = AttentionBackend(backend)
  if backend == AttentionBackend.FLASH_ATTENTION2:
    return _flash_attention2(q, k, v, attn_mask=attn_mask, causal=causal)
  if backend == AttentionBackend.TILELANG:
    return _tilelang_attention(q, k, v, attn_mask=attn_mask, causal=causal, scale=scale)

  return F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=attn_mask,
    dropout_p=0.0,
    is_causal=causal and attn_mask is None,
    scale=scale,
  )


def _flash_attention2(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  *,
  attn_mask: torch.Tensor | None,
  causal: bool,
) -> torch.Tensor:
  if attn_mask is not None:
    raise ValueError("flash_attention2 backend does not support attn_mask")
  try:
    from flash_attn import flash_attn_func
  except ImportError as exc:
    raise ImportError(
      "flash_attention2 backend requires the flash-attn package"
    ) from exc

  q = q.transpose(1, 2).contiguous()
  k = k.transpose(1, 2).contiguous()
  v = v.transpose(1, 2).contiguous()
  value_head_dim = v.shape[-1]
  if value_head_dim != q.shape[-1]:
    v = F.pad(v, [0, q.shape[-1] - value_head_dim])
  y = flash_attn_func(q, k, v, causal=causal)
  return y[..., :value_head_dim].transpose(1, 2).contiguous()


def _tilelang_attention(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  *,
  attn_mask: torch.Tensor | None,
  causal: bool,
  scale: float | None,
) -> torch.Tensor:
  if attn_mask is not None:
    raise ValueError("tilelang backend does not support attn_mask")
  if not causal:
    raise ValueError("tilelang backend currently supports causal attention only")
  if scale is not None and scale != q.shape[-1] ** -0.5:
    raise ValueError("tilelang backend does not support custom attention scale")
  if q.shape != k.shape or q.shape != v.shape:
    raise ValueError("tilelang backend requires q, k, and v to have identical shapes")

  from flm_modules.kernels.tilelang.flash_attention import tilelang_flash_attention

  return tilelang_flash_attention(q, k, v).transpose(1, 2).contiguous()
