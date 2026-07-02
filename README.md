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

## Setup

```sh
uv sync --all-packages
```
