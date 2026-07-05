import json
import math
from pathlib import Path

import numpy as np
import torch
from flm_train.checkpoints import CheckpointState, load_checkpoint, save_checkpoint
from flm_train.data import _token_entropy_nats_from_paths, publish_repo_source_dataset
from flm_train.data_cli import parse_args, run_from_args
from flm_train.presets import train_language_model
from flm_train.trainer import TrainStepMetrics
from flm_train.types import (
  CheckpointConfig,
  DataConfig,
  DeepSeekV4ModelConfig,
  DSTinyModelConfig,
  LoopConfig,
  ModelConfig,
  ReferenceModelConfig,
  TrainConfig,
)


def train_config(
  *,
  repo_root: Path,
  model: ModelConfig | None = None,
  steps: int = 1,
) -> TrainConfig:
  dataset_root = repo_root / ".cache" / "data" / "repo_sources"
  publish_repo_source_dataset(
    repo_root=repo_root,
    dataset_root=dataset_root,
    train_ratio=1.0,
    val_ratio=0.0,
    test_ratio=0.0,
  )
  model_config = model
  if model_config is None:
    model_config = ReferenceModelConfig(
      d_model=8,
      n_layers=1,
      n_heads=2,
      d_ff=16,
    )
  return TrainConfig(
    data=DataConfig(dataset_root=dataset_root, seq_len=8),
    model=model_config,
    loop=LoopConfig(batch_size=2, steps=steps),
  )


def test_train_language_model_runs_one_step(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_language_model(
    train_config(
      repo_root=tmp_path,
    )
  )

  assert result.file_count == 1
  assert result.token_count > result.file_count
  assert len(result.losses) == 1
  assert result.losses[0] > 0


def test_publish_repo_source_dataset_writes_versioned_artifacts(tmp_path: Path) -> None:
  repo_root = tmp_path / "repo"
  dataset_root = tmp_path / "datasets" / "repo_sources"
  repo_root.mkdir()
  (repo_root / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  published = publish_repo_source_dataset(
    repo_root=repo_root,
    dataset_root=dataset_root,
    train_ratio=1.0,
    val_ratio=0.0,
    test_ratio=0.0,
  )

  assert (dataset_root / "latest.json").is_file()
  assert published.manifest_path.is_file()
  assert published.split_paths["train"].is_file()
  assert published.split_paths["val"].is_file()
  assert published.split_paths["test"].is_file()
  assert (published.manifest_path.parent / "files.jsonl").is_file()
  manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))
  assert manifest["split"]["strategy"] == "file_hash"
  assert manifest["unigram_entropy_nats_per_token"] > 0
  assert set(manifest["splits"]) == {"train", "val", "test"}
  assert published.file_count == 1
  assert published.token_count > 0
  assert published.byte_count > 0
  assert manifest["byte_count"] == published.byte_count
  assert manifest["splits"]["train"]["byte_count"] == published.byte_count
  assert (
    published.unigram_entropy_nats_per_token
    == manifest["unigram_entropy_nats_per_token"]
  )


def test_token_entropy_counts_emitted_token_ids_once(tmp_path: Path) -> None:
  tokens_path = tmp_path / "tokens.npy"
  np.save(tokens_path, np.asarray([1, 2, 2, 3], dtype=np.int32))

  entropy = _token_entropy_nats_from_paths([tokens_path])

  expected = -(0.25 * math.log(0.25) + 0.5 * math.log(0.5) + 0.25 * math.log(0.25))
  assert entropy == expected


def test_train_on_published_token_dataset_uses_latest_version(tmp_path: Path) -> None:
  repo_root = tmp_path / "repo"
  dataset_root = tmp_path / "datasets" / "repo_sources"
  repo_root.mkdir()
  (repo_root / "model.py").write_text(
    "\n".join(f"def published_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )
  published = publish_repo_source_dataset(
    repo_root=repo_root,
    dataset_root=dataset_root,
    train_ratio=1.0,
    val_ratio=0.0,
    test_ratio=0.0,
  )

  result = train_language_model(
    TrainConfig(
      data=DataConfig(
        kind="token_dataset",
        dataset_root=dataset_root,
        version="latest",
        split="train",
        encoding_name="cl100k_base",
        seq_len=8,
      ),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=1),
    )
  )

  assert published.version
  assert result.file_count == 1
  assert result.byte_count > 0
  assert result.token_count > 0
  assert result.token_count <= published.token_count
  assert len(result.losses) == 1


def test_data_cli_publishes_repo_sources(tmp_path: Path, capsys) -> None:
  repo_root = tmp_path / "repo"
  dataset_root = tmp_path / "datasets" / "repo_sources"
  repo_root.mkdir()
  (repo_root / "model.py").write_text("x = 1\n", encoding="utf-8")

  args = parse_args(
    [
      "repo-sources",
      "publish",
      "--repo-root",
      str(repo_root),
      "--dataset-root",
      str(dataset_root),
      "--train-ratio",
      "1.0",
      "--val-ratio",
      "0.0",
      "--test-ratio",
      "0.0",
    ]
  )
  run_from_args(args)

  output = capsys.readouterr().out
  assert "version=" in output
  assert "tokens=" in output
  assert "bytes=" in output
  assert "unigram_entropy_nats_per_token=" in output
  assert "train_tokens=" in output
  assert "train_bytes=" in output
  assert "val_tokens=" in output
  assert "test_tokens=" in output
  assert (dataset_root / "latest.json").is_file()


def test_data_cli_trains_unitoken_tokenizer_for_repo_sources(
  tmp_path: Path,
  capsys,
) -> None:
  repo_root = tmp_path / "repo"
  dataset_root = tmp_path / "datasets" / "repo_sources"
  tokenizer_root = tmp_path / "tokenizers"
  repo_root.mkdir()
  (repo_root / "model.py").write_text(
    "\n".join(f"def token_{i}(): return {i}" for i in range(120)),
    encoding="utf-8",
  )

  args = parse_args(
    [
      "repo-sources",
      "publish",
      "--repo-root",
      str(repo_root),
      "--dataset-root",
      str(dataset_root),
      "--unitoken-vocab-size",
      "300",
      "--unitoken-special-token-count",
      "16",
      "--tokenizer-root",
      str(tokenizer_root),
      "--tokenizer-name",
      "repo_300",
      "--train-ratio",
      "1.0",
      "--val-ratio",
      "0.0",
      "--test-ratio",
      "0.0",
    ]
  )
  run_from_args(args)

  output = capsys.readouterr().out
  latest = json.loads((dataset_root / "latest.json").read_text(encoding="utf-8"))
  manifest_path = dataset_root / latest["manifest"]
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

  assert "tokens=" in output
  assert "bytes=" in output
  assert "unigram_entropy_nats_per_token=" in output
  assert manifest["encoding_name"] == f"unitoken:{tokenizer_root / 'repo_300'}"
  assert manifest["unigram_entropy_nats_per_token"] > 0
  assert (tokenizer_root / "vocab.repo_300[u8].json").is_file()
  assert (tokenizer_root / "merges.repo_300[u8].txt").is_file()


def test_train_language_model_emits_step_metrics(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )
  step_metrics: list[TrainStepMetrics] = []

  result = train_language_model(
    train_config(
      repo_root=tmp_path,
      steps=2,
    ),
    on_step=step_metrics.append,
  )

  assert len(step_metrics) == 2
  assert [metrics.step for metrics in step_metrics] == [1, 2]
  assert [metrics.loss for metrics in step_metrics] == result.losses
  assert all(metrics.learning_rate == 3e-4 for metrics in step_metrics)
  assert all(metrics.tokens == 16 for metrics in step_metrics)
  assert [metrics.tokens_seen for metrics in step_metrics] == [16, 32]
  assert all(metrics.grad_norm > 0 for metrics in step_metrics)
  assert all(metrics.bits_per_byte > 0 for metrics in step_metrics)
  assert all(metrics.step_time_sec > 0 for metrics in step_metrics)
  assert all(metrics.tokens_per_sec > 0 for metrics in step_metrics)
  assert "train/loss" in step_metrics[0].to_log_dict()
  assert "train/bpb" in step_metrics[0].to_log_dict()
  assert "train/grad_norm" in step_metrics[0].to_log_dict()
  assert "train/tokens_seen" in step_metrics[0].to_log_dict()


def test_train_language_model_resumes_from_checkpoint(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )
  checkpoint_dir = tmp_path / "checkpoints"

  train_language_model(
    train_config(repo_root=tmp_path, steps=1),
    checkpoint_dir=checkpoint_dir,
  )
  first_metrics: list[TrainStepMetrics] = []
  train_language_model(
    TrainConfig(
      data=DataConfig(
        dataset_root=tmp_path / ".cache" / "data" / "repo_sources",
        seq_len=8,
      ),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=1),
      checkpoint=CheckpointConfig(enabled=True, every_steps=1, keep_last=2),
    ),
    checkpoint_dir=checkpoint_dir,
    on_step=first_metrics.append,
  )
  assert (checkpoint_dir / "step-00000001" / "model.npz").is_file()
  model_state = json.loads(
    (checkpoint_dir / "step-00000001" / "model_state.json").read_text(encoding="utf-8")
  )
  first_tensor = next(
    value["__tensor__"]
    for value in model_state.values()
    if isinstance(value, dict) and "__tensor__" in value
  )
  with np.load(checkpoint_dir / "step-00000001" / "model.npz") as arrays:
    assert first_tensor["name"] in arrays.files
  assert first_tensor["name"] != "tensor_0"
  assert first_tensor["dtype"].startswith("torch.")
  assert first_tensor["shape"]
  assert first_tensor["device"] == "cpu"

  resumed_metrics: list[TrainStepMetrics] = []
  train_language_model(
    TrainConfig(
      data=DataConfig(
        dataset_root=tmp_path / ".cache" / "data" / "repo_sources",
        seq_len=8,
      ),
      model=ReferenceModelConfig(d_model=8, n_layers=1, n_heads=2, d_ff=16),
      loop=LoopConfig(batch_size=2, steps=2),
      checkpoint=CheckpointConfig(
        enabled=True,
        every_steps=1,
        keep_last=2,
        resume="auto",
      ),
    ),
    checkpoint_dir=checkpoint_dir,
    on_step=resumed_metrics.append,
  )

  assert [metrics.step for metrics in first_metrics] == [1]
  assert [metrics.step for metrics in resumed_metrics] == [2]
  assert (checkpoint_dir / "step-00000002" / "model.npz").is_file()


def test_checkpoint_preserves_bfloat16_model_state(tmp_path: Path) -> None:
  model = torch.nn.Linear(4, 2, bias=False).to(dtype=torch.bfloat16)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  expected_weight = model.weight.detach().clone()

  checkpoint = save_checkpoint(
    checkpoint_dir=tmp_path / "checkpoints",
    model=model,
    optimizer=optimizer,
    state=CheckpointState(step=1, tokens_seen=8),
  )
  metadata = json.loads((checkpoint / "model_state.json").read_text(encoding="utf-8"))
  tensor_metadata = metadata["weight"]["__tensor__"]

  assert tensor_metadata["dtype"] == "torch.bfloat16"
  assert tensor_metadata["storage_dtype"] == "uint16"
  with np.load(checkpoint / "model.npz") as arrays:
    assert arrays[tensor_metadata["name"]].dtype == np.uint16

  restored = torch.nn.Linear(4, 2, bias=False).to(dtype=torch.bfloat16)
  restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
  state = load_checkpoint(
    path=checkpoint,
    model=restored,
    optimizer=restored_optimizer,
    map_location="cpu",
  )

  assert state == CheckpointState(step=1, tokens_seen=8)
  assert restored.weight.dtype == torch.bfloat16
  assert torch.equal(restored.weight, expected_weight)


def test_train_language_model_smoke_trains_deepseek_v4(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def g_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_language_model(
    train_config(
      repo_root=tmp_path,
      model=DeepSeekV4ModelConfig(
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=16,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        n_experts_per_token=2,
        n_group=2,
        topk_group=1,
        dense_layers=1,
      ),
    )
  )

  assert result.file_count == 1
  assert result.token_count > result.file_count
  assert len(result.losses) == 1
  assert result.losses[0] > 0


def test_train_language_model_smoke_trains_ds_tiny(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def tiny_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_language_model(
    train_config(
      repo_root=tmp_path,
      model=DSTinyModelConfig(
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=16,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
      ),
    )
  )

  assert result.file_count == 1
  assert result.token_count > result.file_count
  assert len(result.losses) == 1
  assert result.losses[0] > 0


def test_train_language_model_smoke_trains_compressed_deepseek_v4(
  tmp_path: Path,
) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def h_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_language_model(
    train_config(
      repo_root=tmp_path,
      model=DeepSeekV4ModelConfig(
        d_model=16,
        n_layers=2,
        n_heads=2,
        head_dim=8,
        d_ff=16,
        q_lora_rank=8,
        rope_head_dim=8,
        o_lora_rank=4,
        o_groups=2,
        attention_layer_types=(
          "compressed_sparse_attention",
          "heavily_compressed_attention",
        ),
        compress_rate_csa=2,
        compress_rate_hca=2,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        n_routed_experts=4,
        n_shared_experts=1,
        n_experts_per_token=2,
        n_group=2,
        topk_group=1,
        dense_layers=1,
      ),
    )
  )

  assert result.file_count == 1
  assert result.token_count > result.file_count
  assert len(result.losses) == 1
  assert result.losses[0] > 0
