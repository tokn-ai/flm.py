import importlib.util

import pytest
import torch
from flm_modules import AttentionBackend, CausalSelfAttention
from torch.nn import functional as F

try:
  from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
  SDPBackend = None
  sdpa_kernel = None


def test_causal_self_attention_preserves_input_shape(
  random_input,
) -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2)
  x = random_input(3, 5, 8)

  y = layer(x)

  assert y.shape == x.shape


def test_causal_self_attention_accepts_backend_string() -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2, backend="torch")

  assert layer.backend == AttentionBackend.TORCH


def test_causal_self_attention_accepts_tilelang_backend_string() -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2, backend="tilelang")

  assert layer.backend == AttentionBackend.TILELANG


def test_causal_self_attention_matches_saved_output(random_input) -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2)
  x = random_input(3, 5, 8)

  y = layer(x)

  torch.testing.assert_close(
    y[0, 0],
    torch.tensor(
      [
        -0.15856203436851501,
        -0.2611512839794159,
        -0.48171690106391907,
        -0.2562934458255768,
        -0.48398783802986145,
        -0.5388339161872864,
        0.24903729557991028,
        0.2406696379184723,
      ]
    ),
  )


def test_causal_self_attention_matches_scaled_dot_product_attention(
  random_input,
) -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2)
  x = random_input(3, 5, 8)
  batch_size, seq_len, _ = x.shape

  q, k, v = layer.qkv(x).chunk(3, dim=-1)
  q = q.view(batch_size, seq_len, layer.n_heads, layer.head_dim).transpose(1, 2)
  k = k.view(batch_size, seq_len, layer.n_heads, layer.head_dim).transpose(1, 2)
  v = v.view(batch_size, seq_len, layer.n_heads, layer.head_dim).transpose(1, 2)
  q, k = layer.rope(q, k)
  expected = F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
  )
  expected = (
    expected.transpose(1, 2)
    .contiguous()
    .view(
      batch_size,
      seq_len,
      layer.d_model,
    )
  )
  expected = layer.out(expected)

  torch.testing.assert_close(layer(x), expected)


def test_causal_self_attention_rejects_unknown_backend() -> None:
  with pytest.raises(ValueError):
    CausalSelfAttention(d_model=8, n_heads=2, backend="unknown")


def test_causal_self_attention_flash_attention2_requires_package(
  random_input,
) -> None:
  if importlib.util.find_spec("flash_attn") is not None:
    pytest.skip("flash-attn installed; package-missing path not applicable")

  layer = CausalSelfAttention(
    d_model=8,
    n_heads=2,
    backend=AttentionBackend.FLASH_ATTENTION2,
  )
  x = random_input(3, 5, 8)

  with pytest.raises(ImportError, match="flash-attn package"):
    layer(x)


def test_causal_self_attention_tilelang_requires_cuda(random_input) -> None:
  layer = CausalSelfAttention(
    d_model=8,
    n_heads=2,
    backend=AttentionBackend.TILELANG,
  )
  x = random_input(3, 5, 8)

  with pytest.raises(RuntimeError, match="CUDA tensors"):
    layer(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_causal_self_attention_matches_flash_attention_backend(random_input) -> None:
  if sdpa_kernel is None or SDPBackend is None:
    pytest.skip("SDPA backend controls unavailable")

  layer = CausalSelfAttention(d_model=8, n_heads=2).cuda().half()
  x = random_input(3, 5, 8).cuda().half()
  layer.eval()

  expected = layer(x)
  try:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
      actual = layer(x)
  except RuntimeError as exc:
    pytest.skip(f"FlashAttention backend unavailable: {exc}")

  torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_causal_self_attention_matches_flash_attention2_backend(random_input) -> None:
  pytest.importorskip("flash_attn", reason="flash-attn unavailable")

  torch_layer = (
    CausalSelfAttention(
      d_model=8,
      n_heads=2,
      backend=AttentionBackend.TORCH,
    )
    .cuda()
    .half()
  )
  flash_layer = (
    CausalSelfAttention(
      d_model=8,
      n_heads=2,
      backend=AttentionBackend.FLASH_ATTENTION2,
    )
    .cuda()
    .half()
  )
  flash_layer.load_state_dict(torch_layer.state_dict())
  x = random_input(3, 5, 8).cuda().half()

  torch.testing.assert_close(
    flash_layer(x),
    torch_layer(x),
    rtol=1e-3,
    atol=1e-3,
  )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_causal_self_attention_matches_tilelang_backend(random_input) -> None:
  pytest.importorskip("tilelang", reason="TileLang unavailable")

  torch_layer = (
    CausalSelfAttention(
      d_model=8,
      n_heads=2,
      backend=AttentionBackend.TORCH,
    )
    .cuda()
    .half()
  )
  tilelang_layer = (
    CausalSelfAttention(
      d_model=8,
      n_heads=2,
      backend=AttentionBackend.TILELANG,
    )
    .cuda()
    .half()
  )
  tilelang_layer.load_state_dict(torch_layer.state_dict())
  x = random_input(3, 5, 8).cuda().half()

  torch.testing.assert_close(
    tilelang_layer(x),
    torch_layer(x),
    rtol=1e-2,
    atol=1e-2,
  )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_causal_self_attention_tilelang_matches_torch_gradients(
  random_input,
) -> None:
  pytest.importorskip("tilelang", reason="TileLang unavailable")

  torch_layer = (
    CausalSelfAttention(
      d_model=8,
      n_heads=2,
      backend=AttentionBackend.TORCH,
    )
    .cuda()
    .half()
  )
  tilelang_layer = (
    CausalSelfAttention(
      d_model=8,
      n_heads=2,
      backend=AttentionBackend.TILELANG,
    )
    .cuda()
    .half()
  )
  tilelang_layer.load_state_dict(torch_layer.state_dict())
  torch_x = random_input(3, 5, 8).cuda().half().requires_grad_()
  tilelang_x = torch_x.detach().clone().requires_grad_()

  torch_loss = torch_layer(torch_x).float().square().mean()
  tilelang_loss = tilelang_layer(tilelang_x).float().square().mean()
  torch_loss.backward()
  tilelang_loss.backward()

  torch.testing.assert_close(tilelang_x.grad, torch_x.grad, rtol=3e-2, atol=3e-2)
  for tilelang_param, torch_param in zip(
    tilelang_layer.parameters(),
    torch_layer.parameters(),
    strict=True,
  ):
    torch.testing.assert_close(
      tilelang_param.grad,
      torch_param.grad,
      rtol=3e-2,
      atol=3e-2,
    )


def test_causal_self_attention_supports_bias_variant() -> None:
  layer = CausalSelfAttention(d_model=8, n_heads=2, bias=True)

  assert layer.qkv.bias is not None
  assert layer.out.bias is not None


def test_causal_self_attention_rejects_invalid_head_count() -> None:
  with pytest.raises(ValueError, match="d_model must be divisible"):
    CausalSelfAttention(d_model=10, n_heads=3)
