"""Train the reference model on repository source files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from flm_datasets import (
  SourceCorpusConfig,
  TokenDataset,
  encode_text,
  get_tokenizer,
  iter_source_files,
  read_source_corpus,
)
from flm_llm import ReferenceModel, ReferenceModelConfig
from flm_modules import configure_adamw
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
  repo_root: Path = Path(".")
  encoding_name: str = "cl100k_base"
  seq_len: int = 128
  batch_size: int = 8
  steps: int = 10
  learning_rate: float = 3e-4
  weight_decay: float = 0.1
  d_model: int = 128
  n_layers: int = 2
  n_heads: int = 4
  d_ff: int | None = None
  dropout: float = 0.0
  device: str = "cpu"
  seed: int = 42


@dataclass(frozen=True)
class TrainingResult:
  losses: list[float]
  token_count: int
  file_count: int


def train_on_repo_sources(config: TrainConfig) -> TrainingResult:
  torch.manual_seed(config.seed)

  corpus_config = SourceCorpusConfig(root=config.repo_root)
  corpus = read_source_corpus(corpus_config)
  file_count = len(iter_source_files(corpus_config))
  tokens = encode_text(corpus, encoding_name=config.encoding_name)
  dataset = TokenDataset(tokens, seq_len=config.seq_len)
  dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=False,
  )
  encoding = get_tokenizer(config.encoding_name)
  model_config = ReferenceModelConfig(
    vocab_size=encoding.n_vocab,
    max_seq_len=config.seq_len,
    d_model=config.d_model,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    d_ff=config.d_ff,
    dropout=config.dropout,
  )
  model = ReferenceModel(model_config).to(config.device)
  optimizer = configure_adamw(
    model,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
  )

  losses: list[float] = []
  iterator = iter(dataloader)
  model.train()

  for _ in range(config.steps):
    try:
      input_ids, targets = next(iterator)
    except StopIteration:
      iterator = iter(dataloader)
      input_ids, targets = next(iterator)

    input_ids = input_ids.to(config.device)
    targets = targets.to(config.device)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(input_ids, targets)
    if loss is None:
      raise RuntimeError("training loss was not produced")
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))

  return TrainingResult(
    losses=losses,
    token_count=len(tokens),
    file_count=file_count,
  )


def parse_args() -> TrainConfig:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo-root", type=Path, default=Path("."))
  parser.add_argument("--encoding-name", default="cl100k_base")
  parser.add_argument("--seq-len", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--steps", type=int, default=10)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=0.1)
  parser.add_argument("--d-model", type=int, default=128)
  parser.add_argument("--n-layers", type=int, default=2)
  parser.add_argument("--n-heads", type=int, default=4)
  parser.add_argument("--d-ff", type=int, default=None)
  parser.add_argument("--dropout", type=float, default=0.0)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  return TrainConfig(**vars(args))


def main() -> None:
  result = train_on_repo_sources(parse_args())
  for step, loss in enumerate(result.losses, start=1):
    print(f"step={step} loss={loss:.4f}")
  print(f"tokens={result.token_count} files={result.file_count}")
