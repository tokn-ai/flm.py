# Killing the logits tensor: cut cross-entropy for a 16M run

The second entry in the FLM build log. Yesterday was about *shaping* the
training workflow. Today is about one number: **how much GPU memory the loss
eats**, and why the obvious way to compute it spends ~6 GiB when ~25 MiB
suffices.

## Goal

> Train a real **16M-parameter** language model, end to end, with a workflow
> that is reproducible, observable, and easy to extend.

### Task board

- [x] Model architectures in `flm-llm`: ReferenceModel, DSTiny, DeepSeekV4
- [x] Neural building blocks + AdamW optimizer in `flm-modules`
- [x] Dataset loading, incl. CalcQA, in `flm-datasets`
- [x] RL trainers (PPO, GRPO) in `flm-rl`
- [x] Config-driven training: YAML configs, module split, configs-by-kind, local secrets
- [x] Reusable trainer engine and pluggable sinks (files, TensorBoard, MLflow, W&B)
- [ ] (current) Memory-efficient loss backends: torch linear cross-entropy, TileLang CCE, Cut Cross-Entropy
- [ ] Evaluation harness (measured benchmarks, not just train loss)
- [ ] Checkpointing + resume mid-run
- [ ] Real / larger dataset support beyond the repo-source preset

---

## How to use

The loss backend is a YAML knob on the model, and the compute dtype is a knob
on the loop. Switching to a fused, memory-efficient loss is a one-line change:

```yaml
model:
  kind: reference
  loss_backend: cut_cross_entropy   # or linear_cross_entropy / tilelang_linear_cross_entropy
  loss_chunk_size: 512

loop:
  dtype: bfloat16
```

No code changes — the trainer dispatches through `language_model_loss(...)`
[5] in `flm-modules`, and the model never builds a full logits matrix.

## The culprit: a 100k-wide logits tensor

Our tokenizer is `cl100k_base`, so the vocabulary is **V ≈ 100,000**. The
language-modeling head maps every token's hidden state to a distribution over
that vocabulary. The classic recipe is two steps:

```python
logits = F.linear(hidden, classifier_weight)   # [N, V]
loss = F.cross_entropy(logits, targets)
```

where `N` is the number of tokens in the step and `V ≈ 100k`. The problem is
the intermediate `logits`: it is `N × V` elements, and `V` is enormous. At a
training step of `N = 4096` tokens that is ~400 million elements — and
during the backward pass, several tensors of exactly that shape are alive at
once.

## The math: only two numbers per token suffice

Let $h_n \in \mathbb{R}^d$ be the hidden state of token $n$, $w_v \in
\mathbb{R}^d$ the classifier weight for vocabulary entry $v$, and $y_n$ the
target index. The two-step recipe is:

$$
\text{logits}_{n,v} = h_n^\top w_v
$$

$$
L = \frac{1}{N}\sum_n \text{CE}(\text{softmax}(\text{logits}_n),\, y_n)
  = \frac{1}{N}\sum_n \left[ -\text{logits}_{n,y_n} + \log\sum_v \exp(\text{logits}_{n,v}) \right]
$$

The first term is just the dot product with the single target column; the
second is a `logsumexp` over the vocabulary — a reduction. So per token the
loss needs exactly two scalars derived from the `[N, V]` logits, and we can
fold the recipe into one expression:

$$
L = \frac{1}{N}\sum_n \left[ \operatorname{logsumexp}_v(h_n^\top w_v) - h_n^\top w_{y_n} \right]
$$

Neither requires the full matrix to exist at once. We can compute the dot
products column-block by column-block, fold each block into a running
`logsumexp` [2], and discard it — keeping only a `[chunk, V]` sliver plus a
per-token accumulator. That is exactly the fusion CCE performs.

## Why the old path eats ~6 GiB

At training scale — batch 8 × seq 512 — a step carries **N = 4096 tokens**
against **V ≈ 100,000**. A single `[N, V]` tensor in fp32 is then:

```
4096 × 100,000 × 4 B ≈ 1.5 GiB
```

`F.cross_entropy` is numerically stable: it **upcasts its input from bf16 to
fp32** before computing the log-softmax, so the loss region runs at 4 bytes
per element even when the rest of the network is in bf16. Walk the autograd
graph for `F.linear` + `F.cross_entropy` and you find roughly four full
`[N, V]` tensors resident through the backward pass:

- the saved `logits` (bf16),
- the fp32 upcast copy,
- the softmax probabilities (fp32),
- the `grad_logits` (`softmax − onehot`, fp32).

Four blocks at ~1.5 GiB apiece:

```
4 × 1.5 GiB ≈ 6 GiB
```

That is the backward peak of the loss region alone: **~6 GiB**, almost
entirely spent on tensors that exist only to be immediately reduced into a
single scalar per token. It scales with `N × V`, so doubling the batch or
growing the vocabulary adds another ~1.5 GiB per block. On a small GPU aiming
for a 16M run, this is the difference between fitting and OOM.

## Why CCE drops the loss region to ~25 MiB

The fix is to **never materialize the logits**. Cross-entropy is a reduction:
all we need is, per token, the logsumexp over the vocabulary and the logit of
the target. Cut Cross-Entropy (CCE) fuses the matmul and the loss into one
kernel that streams over the vocabulary in chunks, accumulating a small
per-token `logsumexp` instead of the full `[N, V]` matrix.

What stays resident in the loss region:

- `hidden` `[N, d]` — small (`d` is a few hundred).
- `classifier_weight` `[V, d]` — the tied embedding; always resident anyway,
  not extra cost.
- a per-token `logsumexp` `[N]` — a single vector, negligible.
- one working chunk of logits `[chunk, V]` in fp32 — the only real allocation.

With a 64-row chunk that sliver is:

```
64 × 100,000 × 4 B ≈ 24.5 MiB
```

So the loss region collapses from **~6 GiB to ~25 MiB** — roughly a 240x
cut. The fp32 upcast still happens, but only on that 64-row sliver, not the
whole `[N, V]`. Same loss value, same gradients (we test for both).

The total training-step peak — model weights, gradients, optimizer states,
activations, and the loss region together — lands around **~400 MiB**, of
which the loss is now a rounding error. Before, the loss region alone was
~6 GiB and swamped everything else.

## The same trick, twice: Flash Attention

This is not a new idea — it is the same move Flash Attention [1] made for the
attention layer. There the wide intermediate is the attention score matrix
`[N, N]` (sequence × sequence), which exists only to be softmaxed and
reduced into a weighted sum over the values. Flash Attention fuses
`QK^T → softmax → ·V` into one tiled kernel that never writes the full
`[N, N]` scores to HBM, streaming the softmax in blocks and keeping only a
running reduction. The memory win is identical in spirit: kill the
materialized intermediate, keep the reduction.

The two are duals inside FLM:

- **Flash Attention** tames the `[N, N]` sequence intermediate — its cost is
  `seq²`, so it dominates at long context.
- **CCE** tames the `[N, V]` vocabulary intermediate — its cost is `N × V`,
  so at a 100k vocab it dominates even at modest context.

Which monster is bigger depends purely on shape. FLM wires both behind the
same pattern: a `*_backend` YAML knob selecting between a torch reference and
a fused kernel — `attention_backend` (`torch` / `flash_attention2` /
`tilelang`) and `loss_backend` (`cross_entropy` / `linear_cross_entropy` /
`tilelang_linear_cross_entropy` / `cut_cross_entropy`). Same diagnosis, same
cure, two places.

## Backends shipped

Today `flm-modules` exposes four interchangeable backends, all reached
through one dispatch function and verified against the reference
`F.linear + F.cross_entropy` for both forward value *and* backward gradients:

- `cross_entropy` — the reference: materialize logits, then reduce.
- `linear_cross_entropy` — pure-torch chunked CCE; the compatibility fallback,
  no extra deps.
- `tilelang_linear_cross_entropy` — a TileLang CUDA kernel [4], our own fused
  forward/backward with shape-keyed kernels.
- `cut_cross_entropy` — the upstream Cut Cross-Entropy package [3], the
  production pick when available.

Picking one is the YAML line above; the model code does not change.

## Why it mattered

The loss is the last, widest layer of a language model, and at a 100k
vocabulary it is where memory goes to die. By fusing the matmul and the
reduction we cut the loss region from ~6 GiB to ~25 MiB — bringing the whole
training step down to a ~400 MiB peak — without touching the math. That
headroom is what makes a real 16M run fit on the hardware we have, and it is
a prerequisite for the next items on the board: bigger batches, longer
context, and the datasets to feed them.

## References

[1] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. *FlashAttention: Fast and
    Memory-Efficient Exact Attention with IO-Awareness.* In Advances in
    Neural Information Processing Systems (NeurIPS), 2022.
    [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

[2] M. Milakov and N. Gimelshein. *On online normalizer calculation for
    softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867), 2018.

[3] Cut Cross-Entropy (CCE).
    [arXiv:2411.09009](https://arxiv.org/abs/2411.09009), 2024.

[4] TileLang. *TileLang: A DSL for high-performance GPU kernels.*
    <https://github.com/tile-ai/tilelang>.

[5] FLM. *Loss backend dispatch.*
    `packages/modules/src/flm_modules/losses.py`, this repository.
