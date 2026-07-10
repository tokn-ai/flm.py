# FLM

Placeholder `uv` monorepo for LLM experiments.

## Packages

- `flm-llm`: core model code
- `flm-modules`: reusable neural network building blocks
- `flm-datasets`: dataset loading and preprocessing
- `flm-train`: training workflows
- `flm-rl`: reinforcement learning workflows
- `flm-inference`: inference and serving workflows

## Reference Model

`flm-llm` includes a decoder-only reference model using RoPE, causal
attention, SwiGLU feed-forward blocks, RMSNorm, tied token embeddings, and a
cross-entropy language modeling loss. `flm-modules` exposes the reusable
building blocks and an AdamW optimizer helper.

## Training

Run an experiment from a YAML config:

```sh
uv run flm-train-experiment experiments/16m_repo.yaml
```

### Portable nanoGPT speedrun baseline

The eager-PyTorch baseline follows the current short-track architecture,
optimizer recipe, training stages, canonical FineWeb token stream, and exact
10,485,760-token validation target. Download the official pre-tokenized shards
and run it with:

```sh
uv run python scripts/download_nanogpt_speedrun_data.py --run
uv run flm-train-experiment experiments/nanogpt_speedrun.yaml
```

This is the portable semantic baseline. Distributed execution, FP8,
FlashAttention 3, and the fused H100 kernels required for sub-90-second timing
are intentionally outside this configuration.

### 16M FineWeb speedrun-style experiment

The 16M experiment keeps the current speedrun topology and optimizer while
scaling it to the existing 8,192-token Unitoken FineWeb dataset. It has
15,931,066 trainable parameters and a 37,048,320-token staged training budget.
Six-way gradient accumulation preserves that effective budget while keeping
the CUDA microbatches within an 8 GiB GPU. File and TensorBoard logging are
enabled for both the smoke and full runs.

Run the two-step launch smoke test first:

```sh
scripts/smoke_16m_fineweb_speedrun.sh
```

Then start or resume the full experiment:

```sh
scripts/run_16m_fineweb_speedrun.sh
```

Both scripts forward extra CLI options, for example:

```sh
scripts/run_16m_fineweb_speedrun.sh --workspace-config /path/to/flm.workspace.yaml
```

Workspace-specific directories can live outside the experiment config. A local
`flm.workspace.yaml` in the current directory or one of its parents is loaded
automatically and is ignored by git:

```yaml
# flm.workspace.yaml
dirs:
  code_root: .
  workspace_root: .
workspace:
  runs_dir: runs
  data_dir: data
  tokenizers_dir: tokenizers
  models_dir: models
  cache_dir: cache
```

```sh
uv run flm-train-experiment experiments/16m_repo.yaml
```

Relative data, tokenizer, sink, and secret paths resolve from `workspace_root`.
Relative experiment config paths passed to the CLI resolve from `code_root`.
Run artifacts are written to `workspace_root/workspace.runs_dir/experiment/run_id`.

## Setup

```sh
uv sync --all-packages
```

## vLLM CPU Rollout

Install a CPU-enabled vLLM build for the host platform, then select a
CPU-compatible dtype and backend tuning explicitly:

```sh
uv run flm-vllm-rollout MODEL_DIR \
  --dtype bfloat16 \
  --cpu-kvcache-space 4 \
  --cpu-omp-threads-bind auto \
  --enforce-eager \
  --max-num-batched-tokens 1024 \
  --prompt example="Once upon a time"
```

The rollout command translates the CPU tuning arguments into the environment
variables required internally by vLLM before initializing the engine.
Eager mode avoids a long first-run compilation for short, one-shot rollouts.
