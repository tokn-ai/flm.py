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
