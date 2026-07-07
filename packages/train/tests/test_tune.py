import json
from pathlib import Path

import flm_train.cli
from flm_train.config import (
  ExperimentConfig,
  FilesSinkConfig,
  OutputConfig,
  RunConfig,
  TensorBoardSinkConfig,
)
from flm_train.data import publish_repo_source_dataset
from flm_train.tune import (
  build_nsys_command,
  parse_args,
  parse_profilers,
  prepare_tune_config,
  run_torch_memory_profile,
  run_torch_profile,
)
from flm_train.types import (
  CheckpointConfig,
  DataConfig,
  EvalConfig,
  LoopConfig,
  RolloutConfig,
  RolloutPromptConfig,
)


def test_parse_args_accepts_tune_options() -> None:
  args = parse_args(
    [
      "experiments/16m_repo.yaml",
      "--steps",
      "2",
      "--device",
      "cuda",
      "--root-dir",
      "/tmp/runs",
      "--workspace-config",
      "flm.yaml",
      "--project",
      "course",
      "--code-dir",
      "/repo/flm",
      "--work-dir",
      "/work/flm",
      "--output-root",
      "outputs",
      "--profiler",
      "nsys,torch",
      "--nsys-trace",
      "cuda,nvtx",
      "--include-eval",
    ]
  )

  assert args.config == Path("experiments/16m_repo.yaml")
  assert args.steps == 2
  assert args.device == "cuda"
  assert args.root_dir == Path("/tmp/runs")
  assert args.workspace_config == Path("flm.yaml")
  assert args.project == "course"
  assert args.code_dir == Path("/repo/flm")
  assert args.work_dir == Path("/work/flm")
  assert args.output_root == Path("outputs")
  assert args.profiler == "nsys,torch"
  assert args.memory_trace is True
  assert args.nsys_trace == "cuda,nvtx"
  assert args.include_eval is True
  assert args.include_rollout is False


def test_parse_args_can_disable_memory_trace() -> None:
  args = parse_args(["experiments/16m_repo.yaml", "--no-memory-trace"])

  assert args.memory_trace is False


def test_parse_profilers_accepts_all_and_lists() -> None:
  assert parse_profilers("all") == ("memory", "torch", "nsys")
  assert parse_profilers("nsys,torch") == ("nsys", "torch")
  assert parse_profilers("memory") == ("memory",)
  assert parse_profilers("torch") == ("torch",)


def test_prepare_tune_config_disables_noisy_workflows(tmp_path: Path) -> None:
  dataset_root = publish_fixture_dataset(tmp_path)
  config = prepare_tune_config(
    ExperimentConfig(
      name="tune",
      data=DataConfig(dataset_root=dataset_root, seq_len=8),
      run=RunConfig(id="run-123"),
      loop=LoopConfig(steps=3),
      eval=EvalConfig(every_steps=1),
      rollout=RolloutConfig(
        every_steps=1,
        prompts=(RolloutPromptConfig(name="p", prompt="def f():"),),
      ),
      checkpoint=CheckpointConfig(enabled=True, every_steps=1),
      output=OutputConfig(root_dir=Path("runs")),
      sinks=(TensorBoardSinkConfig(),),
    ),
    include_eval=False,
    include_rollout=False,
    include_checkpoint=False,
  )

  assert config.run.id == "run-123"
  assert config.run.group == "tune"
  assert config.run_dir == Path("runs") / "tune" / "run-123"
  assert config.eval is None
  assert config.rollout is None
  assert config.checkpoint.enabled is False
  assert config.system_metrics.enabled is False
  assert config.sinks == (FilesSinkConfig(),)


def test_prepare_tune_config_marks_generated_run_id(tmp_path: Path) -> None:
  dataset_root = publish_fixture_dataset(tmp_path)
  config = prepare_tune_config(
    ExperimentConfig(
      name="tune",
      data=DataConfig(dataset_root=dataset_root, seq_len=8),
      loop=LoopConfig(steps=3),
      output=OutputConfig(root_dir=Path("runs")),
      sinks=(TensorBoardSinkConfig(),),
    ),
    include_eval=False,
    include_rollout=False,
    include_checkpoint=False,
  )

  assert config.run.id is not None
  assert config.run.id.startswith("tune-")
  assert config.run_dir == Path("runs") / "tune" / config.run.id
  assert config.sinks == (FilesSinkConfig(),)


def publish_fixture_dataset(tmp_path: Path) -> Path:
  repo_root = tmp_path / "repo"
  dataset_root = tmp_path / "datasets" / "repo_sources"
  repo_root.mkdir()
  (repo_root / "model.py").write_text(
    "\n".join(f"def f_{index}(): return {index}" for index in range(80)),
    encoding="utf-8",
  )
  publish_repo_source_dataset(
    repo_root=repo_root,
    dataset_root=dataset_root,
    train_ratio=1.0,
    val_ratio=0.0,
    test_ratio=0.0,
  )
  return dataset_root


def test_build_nsys_command_uses_resolved_config_path() -> None:
  config_path = Path("runs/tune/run-123/tune/nsys/config.resolved.yaml").resolve()
  output_prefix = Path("runs/tune/run-123/tune/nsys/profile").resolve()
  command = build_nsys_command(
    nsys="/usr/bin/nsys",
    config_path=config_path,
    output_prefix=output_prefix,
    trace="cuda,nvtx",
  )

  assert command[:5] == [
    "/usr/bin/nsys",
    "profile",
    "--force-overwrite=true",
    f"--output={output_prefix}",
    "--trace=cuda,nvtx",
  ]
  assert command[-3:] == [
    "-m",
    "flm_train.cli",
    str(config_path),
  ]


def test_cli_module_has_main_entrypoint() -> None:
  assert hasattr(flm_train.cli, "main")
  assert flm_train.cli.__name__ != "__main__"


def test_run_torch_profile_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
  calls = []

  class FakeKeyAverages:
    def table(self, *, sort_by, row_limit):
      calls.append(("table", sort_by, row_limit))
      return "memory table"

  class FakeProfiler:
    def __init__(self, **kwargs) -> None:
      self.kwargs = kwargs

    def __enter__(self):
      calls.append(("enter", self.kwargs))
      return self

    def __exit__(self, exc_type, exc, traceback) -> None:
      calls.append(("exit", exc_type))

    def export_chrome_trace(self, path: str) -> None:
      Path(path).write_text("{}", encoding="utf-8")

    def key_averages(self):
      return FakeKeyAverages()

  def fake_profile(**kwargs):
    return FakeProfiler(**kwargs)

  def fake_run_experiment(config, *, log):
    calls.append(("run", config.run_dir))
    log("trained")

  monkeypatch.setattr("flm_train.tune.profile", fake_profile)
  monkeypatch.setattr("flm_train.tune.run_experiment", fake_run_experiment)
  monkeypatch.setattr("flm_train.tune.torch.cuda.is_available", lambda: False)

  run_torch_profile(
    ExperimentConfig(
      name="tune",
      run=RunConfig(id="run-123", name="tune brisk-signal", group="tune"),
      loop=LoopConfig(device="cpu"),
      output=OutputConfig(root_dir=tmp_path),
    ),
    log=lambda message: calls.append(("log", message)),
  )

  tune_dir = tmp_path / "tune" / "run-123" / "tune" / "torch"
  assert not (tune_dir / "trace.json").exists()
  assert (tune_dir / "memory_table.txt").read_text(encoding="utf-8") == (
    "memory table\n"
  )
  assert (tune_dir / "summary.json").is_file()
  assert ("run", tmp_path / "tune" / "run-123") in calls


def test_run_torch_profile_writes_trace_when_requested(
  tmp_path: Path,
  monkeypatch,
) -> None:
  class FakeKeyAverages:
    def table(self, *, sort_by, row_limit):
      del sort_by, row_limit
      return "memory table"

  class FakeProfiler:
    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc, traceback) -> None:
      del exc_type, exc, traceback

    def export_chrome_trace(self, path: str) -> None:
      Path(path).write_text("{}", encoding="utf-8")

    def key_averages(self):
      return FakeKeyAverages()

  monkeypatch.setattr("flm_train.tune.profile", lambda **kwargs: FakeProfiler())
  monkeypatch.setattr("flm_train.tune.run_experiment", lambda config, *, log: None)
  monkeypatch.setattr("flm_train.tune.torch.cuda.is_available", lambda: False)

  run_torch_profile(
    ExperimentConfig(
      name="tune",
      run=RunConfig(id="run-123"),
      loop=LoopConfig(device="cpu"),
      output=OutputConfig(root_dir=tmp_path),
    ),
    export_trace=True,
    log=lambda message: None,
  )

  assert (tmp_path / "tune" / "run-123" / "tune" / "torch" / "trace.json").is_file()


def test_run_torch_memory_profile_writes_cpu_artifacts(
  tmp_path: Path,
  monkeypatch,
) -> None:
  calls = []

  def fake_run_experiment(config, *, log):
    calls.append(("run", config.run_dir))
    log("trained")

  monkeypatch.setattr("flm_train.tune.run_experiment", fake_run_experiment)
  monkeypatch.setattr("flm_train.tune.torch.cuda.is_available", lambda: False)

  run_torch_memory_profile(
    ExperimentConfig(
      name="tune",
      run=RunConfig(id="run-123"),
      loop=LoopConfig(device="cpu"),
      output=OutputConfig(root_dir=tmp_path),
    ),
    log=lambda message: calls.append(("log", message)),
  )

  tune_dir = tmp_path / "tune" / "run-123" / "tune" / "memory"
  assert (tune_dir / "memory_stats_before.json").is_file()
  assert (tune_dir / "memory_stats_after.json").is_file()
  assert not (tune_dir / "memory_snapshot.json").exists()
  assert not (tune_dir / "memory_summary.txt").exists()
  assert ("run", tmp_path / "tune" / "run-123") in calls


def test_run_torch_memory_profile_records_cuda_trace_by_default(
  tmp_path: Path,
  monkeypatch,
) -> None:
  calls = []

  def fake_record_memory_history(**kwargs):
    calls.append(("record", kwargs))

  def fake_run_experiment(config, *, log):
    calls.append(("run", config.run_dir))
    log("trained")

  monkeypatch.setattr("flm_train.tune.run_experiment", fake_run_experiment)
  monkeypatch.setattr("flm_train.tune.torch.cuda.is_available", lambda: True)
  monkeypatch.setattr("flm_train.tune.torch.cuda.synchronize", lambda: None)
  monkeypatch.setattr(
    "flm_train.tune.torch.cuda.memory.reset_peak_memory_stats", lambda: None
  )
  monkeypatch.setattr(
    "flm_train.tune.torch.cuda.memory.reset_accumulated_memory_stats",
    lambda: None,
  )
  monkeypatch.setattr(
    "flm_train.tune.torch.cuda.memory._record_memory_history",
    fake_record_memory_history,
  )
  monkeypatch.setattr(
    "flm_train.tune.torch.cuda.memory.memory_stats_as_nested_dict",
    lambda: {"allocated_bytes": {"all": {"peak": 12}}},
  )
  monkeypatch.setattr(
    "flm_train.tune.torch.cuda.memory_summary",
    lambda *, device: f"summary for {device}",
  )
  monkeypatch.setattr(
    "flm_train.tune.torch.cuda.memory.memory_snapshot",
    lambda *, include_traces: {"segments": [], "include_traces": include_traces},
  )
  monkeypatch.setattr(
    "flm_train.tune.torch.cuda.memory._dump_snapshot",
    lambda path: (
      calls.append(("dump_snapshot", path)) or Path(path).write_bytes(b"memory-viz")
    ),
  )

  run_torch_memory_profile(
    ExperimentConfig(
      name="tune",
      run=RunConfig(id="run-123"),
      loop=LoopConfig(device="cuda"),
      output=OutputConfig(root_dir=tmp_path),
    ),
    log=lambda message: calls.append(("log", message)),
  )

  tune_dir = tmp_path / "tune" / "run-123" / "tune" / "memory"
  summary = json.loads((tune_dir / "summary.json").read_text(encoding="utf-8"))
  assert (tune_dir / "memory_snapshot.json").is_file()
  assert (tune_dir / "memory_snapshot.pickle").read_bytes() == b"memory-viz"
  assert (tune_dir / "memory_summary.txt").read_text(encoding="utf-8") == (
    "summary for cuda"
  )
  assert summary["memory_viz"] == "https://pytorch.org/memory_viz"
  assert summary["memory_viz_snapshot"] == str(tune_dir / "memory_snapshot.pickle")
  assert summary["memory_viz_snapshot_format"] == "pytorch_cuda_memory_viz_pickle"
  assert calls.index(("dump_snapshot", str(tune_dir / "memory_snapshot.pickle"))) < (
    calls.index(("record", {"enabled": None}))
  )
  assert calls[0] == (
    "record",
    {
      "clear_history": True,
      "context": "all",
      "enabled": "all",
      "stacks": "all",
    },
  )
  assert ("record", {"enabled": None}) in calls
