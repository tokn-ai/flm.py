"""TileLang linear cross-entropy kernel wrapper."""

from functools import lru_cache

import torch


def tilelang_linear_cross_entropy(
  hidden_states: torch.Tensor,
  classifier_weight: torch.Tensor,
  targets: torch.Tensor,
) -> torch.Tensor:
  """Compute mean linear cross entropy with TileLang CUDA kernels.

  This is a correctness-oriented baseline. It avoids materializing the full
  token-vocab logits tensor, but it is not yet optimized like Cut Cross-Entropy.
  """
  _validate_inputs(hidden_states, classifier_weight, targets)
  hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
  labels = targets.reshape(-1).contiguous()
  return _TileLangLinearCrossEntropy.apply(
    hidden, classifier_weight.contiguous(), labels
  )


class _TileLangLinearCrossEntropy(torch.autograd.Function):
  @staticmethod
  def forward(
    ctx,
    hidden: torch.Tensor,
    classifier_weight: torch.Tensor,
    targets: torch.Tensor,
  ) -> torch.Tensor:
    losses, lse = _tilelang_linear_cross_entropy_forward(
      hidden,
      classifier_weight,
      targets,
    )
    ctx.save_for_backward(hidden, classifier_weight, targets, lse)
    return losses.mean()

  @staticmethod
  def backward(ctx, grad_out: torch.Tensor):
    hidden, classifier_weight, targets, lse = ctx.saved_tensors
    grad_hidden, grad_weight = _tilelang_linear_cross_entropy_backward(
      hidden,
      classifier_weight,
      targets,
      lse,
      grad_out.reshape(1).contiguous(),
    )
    return grad_hidden, grad_weight, None


def _tilelang_linear_cross_entropy_forward(
  hidden: torch.Tensor,
  classifier_weight: torch.Tensor,
  targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  token_count, d_model = hidden.shape
  vocab_size = classifier_weight.shape[0]
  kernel = _get_forward_kernel(
    token_count=token_count,
    d_model=d_model,
    vocab_size=vocab_size,
    dtype=str(hidden.dtype).removeprefix("torch."),
  )
  losses, lse = kernel(hidden, classifier_weight, targets)
  return losses, lse


def _tilelang_linear_cross_entropy_backward(
  hidden: torch.Tensor,
  classifier_weight: torch.Tensor,
  targets: torch.Tensor,
  lse: torch.Tensor,
  grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  token_count, d_model = hidden.shape
  vocab_size = classifier_weight.shape[0]
  grad_hidden = torch.zeros_like(hidden)
  grad_weight = torch.zeros_like(classifier_weight)
  kernel = _get_backward_kernel(
    token_count=token_count,
    d_model=d_model,
    vocab_size=vocab_size,
    dtype=str(hidden.dtype).removeprefix("torch."),
  )
  kernel(
    hidden,
    classifier_weight,
    targets,
    lse,
    grad_out,
    grad_hidden,
    grad_weight,
  )
  return grad_hidden, grad_weight


def _validate_inputs(
  hidden_states: torch.Tensor,
  classifier_weight: torch.Tensor,
  targets: torch.Tensor,
) -> None:
  if hidden_states.ndim < 2:
    raise ValueError("hidden_states must have shape (..., d_model)")
  if classifier_weight.ndim != 2:
    raise ValueError("classifier_weight must have shape (vocab_size, d_model)")
  if hidden_states.shape[-1] != classifier_weight.shape[-1]:
    raise ValueError("hidden_states and classifier_weight d_model must match")
  if targets.shape != hidden_states.shape[:-1]:
    raise ValueError("targets must match hidden_states leading dimensions")
  if not hidden_states.is_cuda or not classifier_weight.is_cuda or not targets.is_cuda:
    raise RuntimeError("TileLang linear cross entropy requires CUDA tensors")
  if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
    raise TypeError(
      "TileLang linear cross entropy supports float16, bfloat16, and float32"
    )
  if classifier_weight.dtype != hidden_states.dtype:
    raise TypeError("classifier_weight dtype must match hidden_states dtype")
  if targets.dtype != torch.long:
    raise TypeError("targets must have dtype torch.long")


@lru_cache(maxsize=32)
def _get_forward_kernel(
  token_count: int,
  d_model: int,
  vocab_size: int,
  dtype: str,
):
  try:
    import tilelang
    from tilelang import language as T
  except ImportError as exc:
    raise ImportError("TileLang CCE requires the tilelang package") from exc

  @tilelang.jit(out_idx=[-2, -1], target="cuda")
  def forward_kernel(
    token_count: int,
    d_model: int,
    vocab_size: int,
    dtype: str,
  ):
    def kernel(
      hidden: T.Tensor([token_count, d_model], dtype),
      weight: T.Tensor([vocab_size, d_model], dtype),
      targets: T.Tensor([token_count], "int64"),
      losses: T.Tensor([token_count], "float32"),
      lse: T.Tensor([token_count], "float32"),
    ):
      with T.Kernel(token_count, threads=1) as token:
        max_logit = T.alloc_local([1], "float32")
        logit = T.alloc_local([1], "float32")
        denom = T.alloc_local([1], "float32")
        target_logit = T.alloc_local([1], "float32")

        max_logit[0] = -3.4028234663852886e38
        target_logit[0] = 0.0
        for vocab in T.serial(0, vocab_size):
          logit[0] = 0.0
          for dim in T.serial(0, d_model):
            logit[0] += T.cast(hidden[token, dim], "float32") * T.cast(
              weight[vocab, dim],
              "float32",
            )
          max_logit[0] = T.max(max_logit[0], logit[0])
          if vocab == targets[token]:
            target_logit[0] = logit[0]

        denom[0] = 0.0
        for vocab in T.serial(0, vocab_size):
          logit[0] = 0.0
          for dim in T.serial(0, d_model):
            logit[0] += T.cast(hidden[token, dim], "float32") * T.cast(
              weight[vocab, dim],
              "float32",
            )
          denom[0] += T.exp(logit[0] - max_logit[0])

        lse[token] = T.log(denom[0]) + max_logit[0]
        losses[token] = lse[token] - target_logit[0]

    return T.prim_func(kernel)

  return forward_kernel(token_count, d_model, vocab_size, dtype)


@lru_cache(maxsize=32)
def _get_backward_kernel(
  token_count: int,
  d_model: int,
  vocab_size: int,
  dtype: str,
):
  try:
    import tilelang
    from tilelang import language as T
  except ImportError as exc:
    raise ImportError("TileLang CCE requires the tilelang package") from exc

  grad_scale = 1.0 / token_count

  @tilelang.jit(out_idx=[], target="cuda")
  def backward_kernel(
    token_count: int,
    d_model: int,
    vocab_size: int,
    dtype: str,
  ):
    def kernel(
      hidden: T.Tensor([token_count, d_model], dtype),
      weight: T.Tensor([vocab_size, d_model], dtype),
      targets: T.Tensor([token_count], "int64"),
      lse: T.Tensor([token_count], "float32"),
      grad_out: T.Tensor([1], "float32"),
      grad_hidden: T.Tensor([token_count, d_model], dtype),
      grad_weight: T.Tensor([vocab_size, d_model], dtype),
    ):
      with T.Kernel(token_count, vocab_size, threads=1) as (token, vocab):
        logit = T.alloc_local([1], "float32")
        coeff = T.alloc_local([1], "float32")

        logit[0] = 0.0
        for dim in T.serial(0, d_model):
          logit[0] += T.cast(hidden[token, dim], "float32") * T.cast(
            weight[vocab, dim],
            "float32",
          )
        coeff[0] = T.exp(logit[0] - lse[token])
        if vocab == targets[token]:
          coeff[0] -= 1.0
        coeff[0] *= grad_out[0] * grad_scale

        for dim in T.serial(0, d_model):
          T.atomic_add(
            grad_hidden[token, dim],
            T.cast(coeff[0] * T.cast(weight[vocab, dim], "float32"), dtype),
          )
          T.atomic_add(
            grad_weight[vocab, dim],
            T.cast(coeff[0] * T.cast(hidden[token, dim], "float32"), dtype),
          )

    return T.prim_func(kernel)

  return backward_kernel(token_count, d_model, vocab_size, dtype)
