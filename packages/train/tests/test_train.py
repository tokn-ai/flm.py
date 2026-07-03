import json
from pathlib import Path

from flm_train.data import publish_repo_source_dataset
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
  assert set(manifest["splits"]) == {"train", "val", "test"}
  assert published.file_count == 1
  assert published.token_count > 0


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
  assert "train_tokens=" in output
  assert "val_tokens=" in output
  assert "test_tokens=" in output
  assert (dataset_root / "latest.json").is_file()


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
  assert all(metrics.step_time_sec > 0 for metrics in step_metrics)
  assert all(metrics.tokens_per_sec > 0 for metrics in step_metrics)
  assert "train/loss" in step_metrics[0].to_log_dict()
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
