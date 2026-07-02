"""Training data builders."""

from __future__ import annotations

from dataclasses import dataclass

from flm_datasets import (
  SourceCorpusConfig,
  TokenDataset,
  encode_text,
  iter_source_files,
  read_source_corpus,
)
from torch.utils.data import DataLoader

from flm_train.types import TrainConfig


@dataclass(frozen=True)
class RepoSourceDatasetBundle:
  dataloader: DataLoader
  token_count: int
  file_count: int


def build_repo_source_dataset(config: TrainConfig) -> RepoSourceDatasetBundle:
  corpus_config = SourceCorpusConfig(root=config.data.repo_root)
  corpus = read_source_corpus(corpus_config)
  file_count = len(iter_source_files(corpus_config))
  tokens = encode_text(corpus, encoding_name=config.data.encoding_name)
  dataset = TokenDataset(tokens, seq_len=config.data.seq_len)
  dataloader = DataLoader(
    dataset,
    batch_size=config.loop.batch_size,
    shuffle=True,
    drop_last=False,
  )
  return RepoSourceDatasetBundle(
    dataloader=dataloader,
    token_count=len(tokens),
    file_count=file_count,
  )
