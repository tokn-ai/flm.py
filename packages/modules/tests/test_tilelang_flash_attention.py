import pytest
import torch
from flm_modules.kernels.tilelang.flash_attention import tilelang_flash_attention


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
