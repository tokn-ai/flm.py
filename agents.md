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

## Data Artifacts

Avoid `pickle`/`.pkl` and `torch.load`/`.pt` for caches or data artifacts by
default because they are unsafe to read from untrusted or stale locations.
Exception: tune runs may write PyTorch CUDA memory visualizer snapshots as
`.pickle` because `https://pytorch.org/memory_viz` requires that format. Treat
those files as write-only diagnostics for the external viewer; never load them
inside this repo.

For token and array caches:

- use `.npy` with `allow_pickle=False` for a single array;
- write adjacent JSON metadata for version, dtype, counts, and related file
  names;
- keep generated data caches under `.cache/data`;
- only use `.pt` for model checkpoints when the checkpoint format is
  intentionally part of the training/checkpointing design.

Repo source data is a publishing input, not a training-time `DataConfig` kind.
Publish a versioned token dataset first:

```sh
uv run flm-data repo-sources publish \
  --repo-root . \
  --dataset-root .cache/data/repo_sources
```

Experiments train from `kind: token_dataset` with `version: latest` or a pinned
version and an explicit `split`. Resolved run configs must record the concrete
`resolved_version`.

Repo token datasets use file-level hash splits by default:

- `train`: 0.98
- `val`: 0.01
- `test`: 0.01

Training configs should use `split: train`; evaluation should use `val` or
`test`. Do not create train/test splits inside the training loop.

## Blog

Day-log posts live under `blog/`, named `YYYYMMDD-NN-slug.md` where
`YYYYMMDD` is the date and `NN` is the zero-padded order of the post within
that day (01, 02, ...). `slug` is a short kebab-case description.
Translations live under `blog/cn/` with the same filename, e.g.
`blog/20260703-01-from-script-to-system.md` and
`blog/cn/20260703-01-from-script-to-system.md`.

Chinese translations keep recurring technical entity names in English
(Model, Dataset, Optimizer, Protocol, Registry, Config, loss, sink, run,
trainer) so prose maps back to the code; code identifiers are never
translated. The full term list lives in `blog/cn/glossary.md` — keep it
in sync when new terms enter use. Use the compact form **16M** for
parameter counts (not 万).

Structure (top to bottom):

1. `# Title` — framed around the day's *theme*, not just "day N".
2. One-line opener — what this entry is.
3. `## Goal` — the north-star as a blockquote (`>`), kept stable across
   posts so readers track it.
4. `### Task board` — a single merged checklist with three marker states,
   in order Done -> Current -> Next:
   - `- [x]` Done (shipped, one-liners)
   - `- [ ] (current)` In progress right now (one-liners)
   - `- [ ]` Next, not started (one-liners)
   - All three buckets at the same simple one-liner density; no nested
     bullets in the board. The board evolves between posts: current items
     graduate to done, next items become current.
5. `---` rule separating status from story.
6. Body — the narrative of the day, theme-driven and high-level, not a diff
   dump. Lead with the concept; let commits support it. At most one tiny
   snippet to anchor an idea.
7. Closing — short "why it mattered" + one-line summary.

Style: high-level / why-focused over code-heavy; scope is the day's work
but always tied back to the Goal; never duplicate detail between the board
and the body.

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
