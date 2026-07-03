import pytest
import torch
from flm_modules.kernels.tilelang.flash_attention import tilelang_flash_attention
from torch.nn import functional as F


def test_tilelang_flash_attention_requires_cuda() -> None:
  q = torch.randn(1, 2, 3, 4, dtype=torch.float16)
  k = torch.randn(1, 2, 3, 4, dtype=torch.float16)
  v = torch.randn(1, 2, 3, 4, dtype=torch.float16)

  with pytest.raises(RuntimeError, match="CUDA tensors"):
    tilelang_flash_attention(q, k, v)


def test_tilelang_flash_attention_rejects_mismatched_shapes() -> None:
  q = torch.randn(1, 2, 3, 4, dtype=torch.float16)
  k = torch.randn(1, 2, 3, 4, dtype=torch.float16)
  v = torch.randn(1, 2, 3, 6, dtype=torch.float16)

  with pytest.raises(ValueError, match="identical shapes"):
    tilelang_flash_attention(q, k, v)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tilelang_flash_attention_matches_torch_gradients(dtype: torch.dtype) -> None:
  pytest.importorskip("tilelang", reason="TileLang unavailable")
  torch.manual_seed(42)
  q = torch.randn(1, 2, 4, 4, device="cuda", dtype=dtype)
  k = torch.randn(1, 2, 4, 4, device="cuda", dtype=dtype)
  v = torch.randn(1, 2, 4, 4, device="cuda", dtype=dtype)
  grad = torch.randn(1, 4, 2, 4, device="cuda", dtype=dtype)

  expected_q = q.detach().clone().requires_grad_()
  expected_k = k.detach().clone().requires_grad_()
  expected_v = v.detach().clone().requires_grad_()
  expected = F.scaled_dot_product_attention(
    expected_q,
    expected_k,
    expected_v,
    dropout_p=0.0,
    is_causal=True,
  ).transpose(1, 2)
  expected.backward(grad)

  actual_q = q.detach().clone().requires_grad_()
  actual_k = k.detach().clone().requires_grad_()
  actual_v = v.detach().clone().requires_grad_()
  actual = tilelang_flash_attention(actual_q, actual_k, actual_v)
  actual.backward(grad)

  torch.testing.assert_close(actual_q.grad, expected_q.grad, rtol=2e-2, atol=2e-2)
  torch.testing.assert_close(actual_k.grad, expected_k.grad, rtol=2e-2, atol=2e-2)
  torch.testing.assert_close(actual_v.grad, expected_v.grad, rtol=2e-2, atol=2e-2)
