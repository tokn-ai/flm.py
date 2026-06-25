from pathlib import Path

from flm_train import TrainConfig, train_on_repo_sources


def test_train_on_repo_sources_runs_one_step(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def f_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_on_repo_sources(
    TrainConfig(
      repo_root=tmp_path,
      seq_len=8,
      batch_size=2,
      steps=1,
      d_model=8,
      n_layers=1,
      n_heads=2,
      d_ff=16,
    )
  )

  assert result.file_count == 1
  assert result.token_count > result.file_count
  assert len(result.losses) == 1
  assert result.losses[0] > 0


def test_train_on_repo_sources_smoke_trains_deepseek_v4(tmp_path: Path) -> None:
  (tmp_path / "model.py").write_text(
    "\n".join(f"def g_{i}(): return {i}" for i in range(80)),
    encoding="utf-8",
  )

  result = train_on_repo_sources(
    TrainConfig(
      repo_root=tmp_path,
      model_name="deepseek_v4",
      seq_len=8,
      batch_size=2,
      steps=1,
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
    )
  )

  assert result.file_count == 1
  assert result.token_count > result.file_count
  assert len(result.losses) == 1
  assert result.losses[0] > 0
