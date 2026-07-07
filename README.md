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
