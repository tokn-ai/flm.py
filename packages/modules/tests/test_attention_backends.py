import pytest
import torch
from flm_modules import AttentionBackend, scaled_dot_product_attention
from torch.nn import functional as F


def test_scaled_dot_product_attention_matches_torch(random_input) -> None:
  q = random_input(2, 3, 5, 4)
  k = random_input(2, 3, 5, 4)
  v = random_input(2, 3, 5, 6)

  expected = F.scaled_dot_product_attention(
    q,
    k,
    v,
    dropout_p=0.0,
    is_causal=True,
  )

  torch.testing.assert_close(
    scaled_dot_product_attention(
      q,
      k,
      v,
      backend=AttentionBackend.TORCH,
      causal=True,
    ),
    expected,
  )


def test_scaled_dot_product_attention_rejects_unknown_backend(random_input) -> None:
  q = random_input(2, 3, 5, 4)

  with pytest.raises(ValueError):
    scaled_dot_product_attention(q, q, q, backend="unknown")


def test_scaled_dot_product_attention_tilelang_rejects_noncausal(
  random_input,
) -> None:
  q = random_input(2, 3, 5, 4)

  with pytest.raises(ValueError, match="causal attention only"):
    scaled_dot_product_attention(
      q,
      q,
      q,
      backend=AttentionBackend.TILELANG,
      causal=False,
    )


def test_scaled_dot_product_attention_flash_attention2_rejects_mask(
  random_input,
) -> None:
  q = random_input(2, 3, 5, 4)
  mask = torch.zeros(2, 1, 5, 5)

  with pytest.raises(ValueError, match="does not support attn_mask"):
    scaled_dot_product_attention(
      q,
      q,
      q,
      backend=AttentionBackend.FLASH_ATTENTION2,
      attn_mask=mask,
    )
