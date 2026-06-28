"""Train the reference model on repository source files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from flm_datasets import (
  SourceCorpusConfig,
  TokenDataset,
  encode_text,
  get_tokenizer,
  iter_source_files,
  read_source_corpus,
)
from flm_llm import (
  DeepSeekV4,
  DeepSeekV4Config,
  DSTiny,
  DSTinyConfig,
  ReferenceModel,
  ReferenceModelConfig,
)
from flm_modules import configure_adamw
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
  repo_root: Path = Path(".")
  model_name: Literal["reference", "deepseek_v4", "ds_tiny"] = "reference"
  encoding_name: str = "cl100k_base"
  seq_len: int = 128
  batch_size: int = 8
  steps: int = 10
  learning_rate: float = 3e-4
  weight_decay: float = 0.1
  d_model: int = 128
  n_layers: int = 2
  n_heads: int = 4
  head_dim: int | None = None
  d_ff: int | None = None
  q_lora_rank: int | None = None
  kv_lora_rank: int = 64
  qk_nope_head_dim: int = 16
  qk_rope_head_dim: int = 16
  v_head_dim: int = 32
  rope_head_dim: int | None = None
  o_lora_rank: int | None = None
  o_groups: int = 1
  attention_layer_types: tuple[str, ...] | None = None
  compress_rate_csa: int = 4
  compress_rate_hca: int = 128
  index_n_heads: int = 64
  index_head_dim: int = 128
  index_topk: int = 512
  n_routed_experts: int = 4
  n_shared_experts: int = 1
  n_experts_per_token: int = 2
  n_group: int = 2
  topk_group: int = 1
  dense_layers: int = 1
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
  model = build_model(
    config,
    vocab_size=encoding.n_vocab,
  ).to(config.device)
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


def build_model(config: TrainConfig, vocab_size: int) -> torch.nn.Module:
  if config.model_name == "reference":
    return ReferenceModel(
      ReferenceModelConfig(
        vocab_size=vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
      )
    )
  if config.model_name == "deepseek_v4":
    return DeepSeekV4(
      DeepSeekV4Config(
        vocab_size=vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        rope_head_dim=config.rope_head_dim,
        o_lora_rank=config.o_lora_rank,
        o_groups=config.o_groups,
        attention_layer_types=config.attention_layer_types,
        compress_rate_csa=config.compress_rate_csa,
        compress_rate_hca=config.compress_rate_hca,
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        moe_d_ff=config.d_ff,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_experts_per_token=config.n_experts_per_token,
        n_group=config.n_group,
        topk_group=config.topk_group,
        dense_layers=config.dense_layers,
      )
    )
  if config.model_name == "ds_tiny":
    return DSTiny(
      DSTinyConfig(
        vocab_size=vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        d_ff=config.d_ff,
      )
    )
  raise ValueError(f"unknown model_name: {config.model_name}")


def parse_args() -> TrainConfig:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo-root", type=Path, default=Path("."))
  parser.add_argument(
    "--model-name",
    choices=["reference", "deepseek_v4", "ds_tiny"],
    default="reference",
  )
  parser.add_argument("--encoding-name", default="cl100k_base")
  parser.add_argument("--seq-len", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--steps", type=int, default=10)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=0.1)
  parser.add_argument("--d-model", type=int, default=128)
  parser.add_argument("--n-layers", type=int, default=2)
  parser.add_argument("--n-heads", type=int, default=4)
  parser.add_argument("--head-dim", type=int, default=None)
  parser.add_argument("--d-ff", type=int, default=None)
  parser.add_argument("--q-lora-rank", type=int, default=None)
  parser.add_argument("--kv-lora-rank", type=int, default=64)
  parser.add_argument("--qk-nope-head-dim", type=int, default=16)
  parser.add_argument("--qk-rope-head-dim", type=int, default=16)
  parser.add_argument("--v-head-dim", type=int, default=32)
  parser.add_argument("--rope-head-dim", type=int, default=None)
  parser.add_argument("--o-lora-rank", type=int, default=None)
  parser.add_argument("--o-groups", type=int, default=1)
  parser.add_argument("--attention-layer-types", nargs="*", default=None)
  parser.add_argument("--compress-rate-csa", type=int, default=4)
  parser.add_argument("--compress-rate-hca", type=int, default=128)
  parser.add_argument("--index-n-heads", type=int, default=64)
  parser.add_argument("--index-head-dim", type=int, default=128)
  parser.add_argument("--index-topk", type=int, default=512)
  parser.add_argument("--n-routed-experts", type=int, default=4)
  parser.add_argument("--n-shared-experts", type=int, default=1)
  parser.add_argument("--n-experts-per-token", type=int, default=2)
  parser.add_argument("--n-group", type=int, default=2)
  parser.add_argument("--topk-group", type=int, default=1)
  parser.add_argument("--dense-layers", type=int, default=1)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  values = vars(args)
  if values["attention_layer_types"]:
    values["attention_layer_types"] = tuple(values["attention_layer_types"])
  else:
    values["attention_layer_types"] = None
  return TrainConfig(**values)


def main() -> None:
  result = train_on_repo_sources(parse_args())
  for step, loss in enumerate(result.losses, start=1):
    print(f"step={step} loss={loss:.4f}")
  print(f"tokens={result.token_count} files={result.file_count}")


if __name__ == "__main__":
  main()
