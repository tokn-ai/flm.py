from pathlib import Path

from flm_train.presets import train_on_repo_sources
from flm_train.trainer import TrainStepMetrics
from flm_train.types import (
  DataTrainConfig,
  LoopTrainConfig,
  ModelTrainConfig,
  TrainConfig,
)


def train_config(
  *,
  repo_root: Path,
  model: ModelTrainConfig | None = None,
  steps: int = 1,
) -> TrainConfig:
  model_config = model
  if model_config is None:
    model_config = ModelTrainConfig(
      d_model=8,
      n_layers=1,
      n_heads=2,
      d_ff=16,
    )
  return TrainConfig(
    data=DataTrainConfig(repo_root=repo_root, seq_len=8),
    model=model_config,
    loop=LoopTrainConfig(batch_size=2, steps=steps),
  )


def test_train_on_repo_sources_runs_one_step(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_on_repo_sources(
    train_config(
      repo_root=tmp_path,
    )
  )

  assert result.file_count == 1
  assert result.token_count > result.file_count
  assert len(result.losses) == 1
  assert result.losses[0] > 0


def test_train_on_repo_sources_emits_step_metrics(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )
  step_metrics: list[TrainStepMetrics] = []

  result = train_on_repo_sources(
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


def test_train_on_repo_sources_smoke_trains_deepseek_v4(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def g_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_on_repo_sources(
    train_config(
      repo_root=tmp_path,
      model=ModelTrainConfig(
        kind="deepseek_v4",
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


def test_train_on_repo_sources_smoke_trains_ds_tiny(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def tiny_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_on_repo_sources(
    train_config(
      repo_root=tmp_path,
      model=ModelTrainConfig(
        kind="ds_tiny",
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


def test_train_on_repo_sources_smoke_trains_compressed_deepseek_v4(
  tmp_path: Path,
) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def h_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_on_repo_sources(
    train_config(
      repo_root=tmp_path,
      model=ModelTrainConfig(
        kind="deepseek_v4",
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
