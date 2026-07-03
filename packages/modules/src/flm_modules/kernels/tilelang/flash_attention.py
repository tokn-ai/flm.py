"""TileLang FlashAttention kernel wrapper.

This backend is intentionally optional. TileLang currently pulls in a native TVM
stack, so imports and compilation happen only when the backend is selected.
"""

from functools import lru_cache

import torch


def tilelang_flash_attention(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
) -> torch.Tensor:
  """Run causal attention with a TileLang NVIDIA kernel.

  Inputs use the module-internal layout: ``(batch, heads, seq, head_dim)``.
  The returned tensor uses ``(batch, seq, heads, head_dim)`` so callers can
  flatten it directly back to model dimension.
  """
  _validate_inputs(q, k, v)
  return _TileLangFlashAttention.apply(q, k, v)


class _TileLangFlashAttention(torch.autograd.Function):
  @staticmethod
  def forward(
    ctx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
  ) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out, lse = _tilelang_flash_attention_forward(q, k, v)
    ctx.save_for_backward(q, k, v, out, lse)
    return out

  @staticmethod
  def backward(ctx, grad_out: torch.Tensor):
    q, k, v, out, lse = ctx.saved_tensors
    grad_out = grad_out.contiguous()
    dq, dk, dv = _tilelang_flash_attention_backward(q, k, v, out, lse, grad_out)
    return dq, dk, dv


def _tilelang_flash_attention_forward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  batch_size, n_heads, seq_len, head_dim = q.shape
  kernel = _get_tilelang_kernel(
    batch_size=batch_size,
    n_heads=n_heads,
    seq_len=seq_len,
    head_dim=head_dim,
    dtype=str(q.dtype).removeprefix("torch."),
  )
  return kernel(q, k, v)


def _tilelang_flash_attention_backward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  lse: torch.Tensor,
  grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  batch_size, n_heads, seq_len, head_dim = q.shape
  dtype = str(q.dtype).removeprefix("torch.")
  dq_kernel = _get_tilelang_dq_kernel(
    batch_size=batch_size,
    n_heads=n_heads,
    seq_len=seq_len,
    head_dim=head_dim,
    dtype=dtype,
  )
  dkv_kernel = _get_tilelang_dkv_kernel(
    batch_size=batch_size,
    n_heads=n_heads,
    seq_len=seq_len,
    head_dim=head_dim,
    dtype=dtype,
  )
  dq = dq_kernel(q, k, v, out, lse, grad_out)
  dk, dv = dkv_kernel(q, k, v, out, lse, grad_out)
  return dq, dk, dv


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
  if q.shape != k.shape or q.shape != v.shape:
    raise ValueError("q, k, and v must have identical shapes")
  if q.ndim != 4:
    raise ValueError("q, k, and v must have shape (batch, heads, seq, head_dim)")
  if not q.is_cuda:
    raise RuntimeError("TileLang attention backend requires NVIDIA CUDA tensors")
  if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
    raise TypeError(
      "TileLang attention backend supports float16, bfloat16, and float32"
    )
  if q.shape[-1] > 256:
    raise ValueError("TileLang attention backend currently supports head_dim <= 256")
  if q.shape[-1] % 2 != 0:
    raise ValueError("TileLang attention backend requires an even head_dim")


@lru_cache(maxsize=32)
def _get_tilelang_kernel(
  batch_size: int,
  n_heads: int,
  seq_len: int,
  head_dim: int,
  dtype: str,
):
  try:
    import tilelang
    from tilelang import language as T
  except ImportError as exc:
    raise ImportError(
      "TileLang attention backend requires the tilelang package"
    ) from exc

  scale = head_dim**-0.5

  @tilelang.jit(out_idx=[-2, -1], target="cuda")
  def flash_attention_kernel():
    @T.prim_func
    def tilelang_flash_attention_forward_kernel(
      q: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      k: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      v: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      out: T.Tensor([batch_size, seq_len, n_heads, head_dim], dtype),
      lse: T.Tensor([batch_size, n_heads, seq_len], "float32"),
    ):
      with T.Kernel(batch_size, n_heads, seq_len, threads=1) as (batch, head, row):
        acc = T.alloc_local([head_dim], "float32")
        m = T.alloc_local([1], "float32")
        denom = T.alloc_local([1], "float32")
        score = T.alloc_local([1], "float32")
        m_new = T.alloc_local([1], "float32")
        alpha = T.alloc_local([1], "float32")
        beta = T.alloc_local([1], "float32")

        m[0] = -3.4028234663852886e38
        denom[0] = 0.0
        T.clear(acc)

        for col in T.serial(0, seq_len):
          if col <= row:
            score[0] = 0.0
            for dim in T.serial(0, head_dim):
              score[0] += (
                T.cast(q[batch, head, row, dim], "float32")
                * T.cast(k[batch, head, col, dim], "float32")
                * scale
              )

            m_new[0] = T.max(m[0], score[0])
            alpha[0] = T.exp(m[0] - m_new[0])
            beta[0] = T.exp(score[0] - m_new[0])

            for dim in T.serial(0, head_dim):
              acc[dim] = acc[dim] * alpha[0] + (
                beta[0] * T.cast(v[batch, head, col, dim], "float32")
              )

            denom[0] = denom[0] * alpha[0] + beta[0]
            m[0] = m_new[0]

        for dim in T.serial(0, head_dim):
          out[batch, row, head, dim] = T.cast(acc[dim] / denom[0], dtype)
        lse[batch, head, row] = T.log(denom[0]) + m[0]

    return tilelang_flash_attention_forward_kernel

  return flash_attention_kernel()


@lru_cache(maxsize=32)
def _get_tilelang_dq_kernel(
  batch_size: int,
  n_heads: int,
  seq_len: int,
  head_dim: int,
  dtype: str,
):
  try:
    import tilelang
    from tilelang import language as T
  except ImportError as exc:
    raise ImportError(
      "TileLang attention backend requires the tilelang package"
    ) from exc

  scale = head_dim**-0.5

  @tilelang.jit(out_idx=[-1], target="cuda")
  def dq_kernel():
    @T.prim_func
    def tilelang_flash_attention_dq_kernel(
      q: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      k: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      v: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      out: T.Tensor([batch_size, seq_len, n_heads, head_dim], dtype),
      lse: T.Tensor([batch_size, n_heads, seq_len], "float32"),
      grad_out: T.Tensor([batch_size, seq_len, n_heads, head_dim], dtype),
      dq: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
    ):
      with T.Kernel(batch_size, n_heads, seq_len, threads=1) as (batch, head, row):
        grad = T.alloc_local([head_dim], "float32")
        score = T.alloc_local([1], "float32")
        prob = T.alloc_local([1], "float32")
        d_prob = T.alloc_local([1], "float32")
        delta = T.alloc_local([1], "float32")
        d_score = T.alloc_local([1], "float32")

        T.clear(grad)
        delta[0] = 0.0
        for dim in T.serial(0, head_dim):
          delta[0] += T.cast(grad_out[batch, row, head, dim], "float32") * T.cast(
            out[batch, row, head, dim], "float32"
          )

        for col in T.serial(0, seq_len):
          if col <= row:
            score[0] = 0.0
            d_prob[0] = 0.0
            for dim in T.serial(0, head_dim):
              score[0] += (
                T.cast(q[batch, head, row, dim], "float32")
                * T.cast(k[batch, head, col, dim], "float32")
                * scale
              )
              d_prob[0] += T.cast(grad_out[batch, row, head, dim], "float32") * T.cast(
                v[batch, head, col, dim], "float32"
              )

            prob[0] = T.exp(score[0] - lse[batch, head, row])
            d_score[0] = prob[0] * (d_prob[0] - delta[0])
            for dim in T.serial(0, head_dim):
              grad[dim] += d_score[0] * T.cast(k[batch, head, col, dim], "float32")

        for dim in T.serial(0, head_dim):
          dq[batch, head, row, dim] = T.cast(grad[dim] * scale, dtype)

    return tilelang_flash_attention_dq_kernel

  return dq_kernel()


@lru_cache(maxsize=32)
def _get_tilelang_dkv_kernel(
  batch_size: int,
  n_heads: int,
  seq_len: int,
  head_dim: int,
  dtype: str,
):
  try:
    import tilelang
    from tilelang import language as T
  except ImportError as exc:
    raise ImportError(
      "TileLang attention backend requires the tilelang package"
    ) from exc

  scale = head_dim**-0.5

  @tilelang.jit(out_idx=[-2, -1], target="cuda")
  def dkv_kernel():
    @T.prim_func
    def tilelang_flash_attention_dkv_kernel(
      q: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      k: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      v: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      out: T.Tensor([batch_size, seq_len, n_heads, head_dim], dtype),
      lse: T.Tensor([batch_size, n_heads, seq_len], "float32"),
      grad_out: T.Tensor([batch_size, seq_len, n_heads, head_dim], dtype),
      dk: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
      dv: T.Tensor([batch_size, n_heads, seq_len, head_dim], dtype),
    ):
      with T.Kernel(batch_size, n_heads, seq_len, threads=1) as (batch, head, col):
        grad_k = T.alloc_local([head_dim], "float32")
        grad_v = T.alloc_local([head_dim], "float32")
        score = T.alloc_local([1], "float32")
        prob = T.alloc_local([1], "float32")
        d_prob = T.alloc_local([1], "float32")
        delta = T.alloc_local([1], "float32")
        d_score = T.alloc_local([1], "float32")

        T.clear(grad_k)
        T.clear(grad_v)

        for row in T.serial(0, seq_len):
          if col <= row:
            score[0] = 0.0
            d_prob[0] = 0.0
            delta[0] = 0.0
            for dim in T.serial(0, head_dim):
              score[0] += (
                T.cast(q[batch, head, row, dim], "float32")
                * T.cast(k[batch, head, col, dim], "float32")
                * scale
              )
              d_prob[0] += T.cast(grad_out[batch, row, head, dim], "float32") * T.cast(
                v[batch, head, col, dim], "float32"
              )
              delta[0] += T.cast(grad_out[batch, row, head, dim], "float32") * T.cast(
                out[batch, row, head, dim], "float32"
              )

            prob[0] = T.exp(score[0] - lse[batch, head, row])
            d_score[0] = prob[0] * (d_prob[0] - delta[0])
            for dim in T.serial(0, head_dim):
              grad_k[dim] += d_score[0] * T.cast(q[batch, head, row, dim], "float32")
              grad_v[dim] += prob[0] * T.cast(
                grad_out[batch, row, head, dim], "float32"
              )

        for dim in T.serial(0, head_dim):
          dk[batch, head, col, dim] = T.cast(grad_k[dim] * scale, dtype)
          dv[batch, head, col, dim] = T.cast(grad_v[dim], dtype)

    return tilelang_flash_attention_dkv_kernel

  return dkv_kernel()
