from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from flm_modules.kernels.tilelang.flash_attention import tilelang_flash_attention
from torch.nn import functional as F


def test_tilelang_flash_attention_16m_shape_is_not_catastrophic() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA attention benchmark requires CUDA")
  pytest.importorskip("tilelang", reason="TileLang unavailable")

  torch.manual_seed(0)
  batch_size = 8
  n_heads = 2
  seq_len = 512
  head_dim = 64
  dtype = torch.bfloat16
  q = torch.randn(
    batch_size,
    n_heads,
    seq_len,
    head_dim,
    device="cuda",
    dtype=dtype,
  )
  k = torch.randn_like(q)
  v = torch.randn_like(q)

  torch_forward_ms = _measure_forward_ms(
    lambda: F.scaled_dot_product_attention(
      q,
      k,
      v,
      dropout_p=0.0,
      is_causal=True,
    ).transpose(1, 2),
  )
  tilelang_forward_ms = _measure_forward_ms(lambda: tilelang_flash_attention(q, k, v))
  torch_backward_ms = _measure_backward_ms(
    lambda q, k, v: F.scaled_dot_product_attention(
      q,
      k,
      v,
      dropout_p=0.0,
      is_causal=True,
    ).transpose(1, 2),
    q=q,
    k=k,
    v=v,
  )
  tilelang_backward_ms = _measure_backward_ms(
    tilelang_flash_attention,
    q=q,
    k=k,
    v=v,
  )

  forward_ratio = tilelang_forward_ms / max(torch_forward_ms, 1.0e-12)
  backward_ratio = tilelang_backward_ms / max(torch_backward_ms, 1.0e-12)
  print(
    "tilelang attention 16m shape: "
    f"forward={tilelang_forward_ms:.3f}ms torch_forward={torch_forward_ms:.3f}ms "
    f"forward_ratio={forward_ratio:.1f}x backward={tilelang_backward_ms:.3f}ms "
    f"torch_backward={torch_backward_ms:.3f}ms backward_ratio={backward_ratio:.1f}x"
  )
  assert forward_ratio < 1_000.0
  assert backward_ratio < 1_000.0


def _measure_forward_ms(fn: Callable[[], torch.Tensor]) -> float:
  for _ in range(2):
    fn()
  torch.cuda.synchronize()
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  output = fn()
  end.record()
  torch.cuda.synchronize()
  assert output.numel() > 0
  return float(start.elapsed_time(end))


def _measure_backward_ms(
  fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
  *,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
) -> float:
  grad = torch.randn(
    q.shape[0], q.shape[2], q.shape[1], q.shape[3], device=q.device, dtype=q.dtype
  )
  for _ in range(2):
    measured_q = q.detach().clone().requires_grad_(True)
    measured_k = k.detach().clone().requires_grad_(True)
    measured_v = v.detach().clone().requires_grad_(True)
    fn(measured_q, measured_k, measured_v).backward(grad)
  torch.cuda.synchronize()

  measured_q = q.detach().clone().requires_grad_(True)
  measured_k = k.detach().clone().requires_grad_(True)
  measured_v = v.detach().clone().requires_grad_(True)
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  fn(measured_q, measured_k, measured_v).backward(grad)
  end.record()
  torch.cuda.synchronize()
  return float(start.elapsed_time(end))
