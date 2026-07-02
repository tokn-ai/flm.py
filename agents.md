# Agent Notes

## Goal

We are building FLM toward a real 16M-parameter training run. The immediate
focus is making the training workflow product-shaped before improving model
quality:

- keep experiment configuration in YAML;
- keep config loading, CLI parsing, and run-loop logic split into clear modules;
- make runs reproducible by saving resolved config and result artifacts;
- add evaluation, checkpointing, resume, logging, and real dataset support next.

## uv

Use `uv` for all Python environment, test, lint, and package operations.

Sync the workspace with NVIDIA/kernel extras when preparing the repo:

```sh
uv sync --all-packages --extra nvidia
```

Use `uv run` for commands inside the environment:

```sh
uv run pytest -q
uv run ruff check .
uv run flm-train-experiment experiments/16m_repo.yaml
```

Prefer normal synced commands for final verification.

## Commits

The agent manages commits unless the user says otherwise.

Use conventional commit messages, for example:

```text
feat(train): add YAML experiment runner
refactor(train): split experiment config cli and runner
```

Before committing:

- check `git status --short`;
- stage only files related to the current task;
- run focused tests or full tests depending on scope;
- do not revert unrelated user changes.

## Current Training Entry Point

The current experiment command is:

```sh
uv run flm-train-experiment experiments/16m_repo.yaml
```

For a CPU smoke run:

```sh
uv run flm-train-experiment experiments/16m_repo.yaml \
  --device cpu \
  --steps 1 \
  --run-dir /tmp/flm-experiment-smoke
```
