# From script to system: a day of shaping the training workflow

This is the first entry in a log tracking the build of **FLM**, a small `uv`
monorepo for LLM experiments.

## Goal

> Train a real **16M-parameter** language model, end to end, with a workflow
> that is reproducible, observable, and easy to extend.

### Task board

- [x] Model architectures in `flm-llm`: ReferenceModel, DSTiny, DeepSeekV4
- [x] Neural building blocks + AdamW optimizer in `flm-modules`
- [x] Dataset loading, incl. CalcQA, in `flm-datasets`
- [x] RL trainers (PPO, GRPO) in `flm-rl`
- [ ] (current) Config-driven training: YAML experiment configs, module split, configs-by-kind, local secrets
- [ ] (current) Reusable trainer engine and pluggable sinks (files, TensorBoard, MLflow, W&B)
- [ ] Evaluation harness (measured benchmarks, not just train loss)
- [ ] Checkpointing + resume mid-run
- [ ] Real / larger dataset support beyond the repo-source preset

---

Today's commits were not about model quality at all. They were about turning
a one-off training *script* into a *system* — config-driven, pluggable, and
decoupled. The rest of this post is a high-level tour of what changed and why.

## The starting point

Before today, training lived in two big files: a repo-specific `train.py`
that baked CLI argument parsing, model construction, the training loop, and
metric writing into a single 260-line script, alongside an `experiment.py`
monolith doing the same thing at a different layer. It worked, but it was
hard to grow: every new model, dataset, or logging backend meant editing the
same tangled file, and runs were driven by flags rather than recorded config.

That's the wrong shape for something heading toward a real 16M run. So the
day was spent reshaping it.

## What changed

The throughline of the day is **separation of concerns**. The work breaks
into four threads.

### 1. Config as the source of truth

Runs are now driven by YAML files under `experiments/`, starting with
`16m_repo.yaml`. A run no longer needs a wall of CLI flags — you point the
runner at a config and it resolves everything:

```sh
uv run flm-train-experiment experiments/16m_repo.yaml
```

The old `experiment.py` monolith was split into focused modules: `config.py`
(hierarchy of frozen dataclasses + YAML loading), `cli.py` (thin argument
parsing), and `runner.py` (orchestration). Configuration is data, not code.

### 2. A reusable training engine

The actual step loop was lifted out of the experiment code into
`trainer.py`, a generic engine driven by a small `LanguageModel` protocol
and emitting structured `TrainStepMetrics`. The experiment runner *uses* the
trainer; it does not *contain* it. The old repo-specific `train.py` was
deleted entirely, replaced by thin `data.py` / `models.py` / `presets.py`
helpers. The training loop now has exactly one implementation.

### 3. Pluggable metric sinks

Observability went from one `sinks.py` file to a `sinks/` package built on a
`Sink` protocol and a registry. Four real backends ship today — `files`
(JSON/JSONL artifacts on disk), `TensorBoard`, `MLflow`, and `Weights &
Biases` — and a YAML config simply selects which ones a run uses. Adding a
new backend means adding one module and registering it; nothing else
changes. This is the open/closed principle, earned the small way.

### 4. Config that scales with the project

Model configuration was reorganized to be a discriminated union **by kind**
(`ReferenceModelConfig`, `DSTinyModelConfig`, `DeepSeekV4ModelConfig`), each
tagged with a `Literal` so YAML stays unambiguous as more architectures
arrive. Training configuration was grouped into coherent blocks
(`TrainConfig`, `LoopConfig`, `OptimizerConfig`, `DataConfig`). A small
`secrets.py` handles loading a local `.secret` env file so tokens like
HuggingFace or W&B keys never leak into the repo or the command line.

## Why it mattered

None of today's commits move the loss curve. What they move is the cost of
the next change. When evaluation lands next, it slots into a config and a
sink — not a rewrite. When checkpointing arrives, it goes into a runner that
already separates orchestration from the step loop. The 16M run is the
destination; today was about making sure the road to it is paved.

## Summary

In one day `flm-train` went from a script that happened to train a model to
a small system: config-driven, engine-reusable, sink-pluggable, and
secret-safe. The model itself is untouched. That's the point — the work that
makes the *next* model cheaper to ship.
