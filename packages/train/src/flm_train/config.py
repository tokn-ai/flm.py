"""Experiment configuration types and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from flm_train.types import (
  CheckpointConfig,
  DataConfig,
  DeepSeekV4ModelConfig,
  DSTinyModelConfig,
  EvalConfig,
  LoopConfig,
  ModelConfig,
  NanoGPTSpeedrunModelConfig,
  OptimizerConfig,
  OptimizerScheduleConfig,
  ReferenceModelConfig,
  RolloutConfig,
  RolloutPromptConfig,
  TorchDType,
  TrainConfig,
)

WORKSPACE_CONFIG_NAME = "flm.workspace.yaml"


@dataclass(frozen=True)
class SecretsConfig:
  env_file: Path | None = Path(".secret")


@dataclass(frozen=True)
class OutputConfig:
  root_dir: Path = Path("runs")


@dataclass(frozen=True)
class WorkspaceConfig:
  code_root: Path = Path(".")
  workspace_root: Path = Path(".")
  runs_dir: Path = Path("runs")
  data_dir: Path = Path("data")
  tokenizers_dir: Path = Path("tokenizers")
  models_dir: Path = Path("models")
  cache_dir: Path = Path("cache")

  @property
  def runs_path(self) -> Path:
    return self.workspace_root / self.runs_dir

  @property
  def data_path(self) -> Path:
    return self.workspace_root / self.data_dir

  @property
  def tokenizers_path(self) -> Path:
    return self.workspace_root / self.tokenizers_dir

  @property
  def models_path(self) -> Path:
    return self.workspace_root / self.models_dir

  @property
  def cache_path(self) -> Path:
    return self.workspace_root / self.cache_dir

  def experiment_dir(self, experiment_name: str) -> Path:
    return self.runs_path / experiment_name

  def run_dir(self, experiment_name: str, run_id: str | None) -> Path:
    experiment_dir = self.experiment_dir(experiment_name)
    if run_id is None:
      return experiment_dir
    return experiment_dir / run_id


@dataclass(frozen=True)
class RunConfig:
  id: str | None = None
  name: str | None = None
  group: str | None = None


@dataclass(frozen=True)
class FilesSinkConfig:
  kind: Literal["files"] = "files"
  run_dir: Path | None = None
  config_json: str = "config.json"
  resolved_config_yaml: str = "config.resolved.yaml"
  status_json: str = "status.json"
  metrics_jsonl: str = "metrics.jsonl"
  system_metrics_jsonl: str = "system_metrics.jsonl"
  result_json: str = "result.json"


@dataclass(frozen=True)
class TensorBoardSinkConfig:
  kind: Literal["tensorboard"] = "tensorboard"
  log_dir: Path | None = None
  flush_secs: int = 10


@dataclass(frozen=True)
class MlflowSinkConfig:
  kind: Literal["mlflow"] = "mlflow"
  tracking_uri: str | None = None
  experiment_name: str | None = None
  run_name: str | None = None
  nested: bool = False


@dataclass(frozen=True)
class WandbSinkConfig:
  kind: Literal["wandb"] = "wandb"
  project: str = "flm"
  entity: str | None = None
  name: str | None = None
  mode: str | None = None
  dir: Path | None = None
  tags: tuple[str, ...] = ()
  group: str | None = None
  job_type: str | None = None


@dataclass(frozen=True)
class SystemMetricsConfig:
  enabled: bool = True
  every_seconds: float = 5.0


SinkConfig = (
  FilesSinkConfig | TensorBoardSinkConfig | MlflowSinkConfig | WandbSinkConfig
)


@dataclass(frozen=True)
class ExperimentConfig:
  name: str
  data: DataConfig = field(default_factory=DataConfig)
  model: ModelConfig = field(default_factory=ReferenceModelConfig)
  optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
  schedule: OptimizerScheduleConfig = field(default_factory=OptimizerScheduleConfig)
  loop: LoopConfig = field(default_factory=LoopConfig)
  eval: EvalConfig | None = None
  rollout: RolloutConfig | None = None
  checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
  system_metrics: SystemMetricsConfig = field(default_factory=SystemMetricsConfig)
  run: RunConfig = field(default_factory=RunConfig)
  secrets: SecretsConfig = field(default_factory=SecretsConfig)
  output: OutputConfig = field(default_factory=OutputConfig)
  sinks: tuple[SinkConfig, ...] = field(default_factory=tuple)

  @property
  def run_dir(self) -> Path:
    if self.run.id is None:
      return self.output.root_dir / self.name
    return self.output.root_dir / self.name / self.run.id

  def to_train_config(self) -> TrainConfig:
    if self.data.kind != "token_dataset":
      raise ValueError(f"unsupported data.kind: {self.data.kind}")
    if self.optimizer.kind not in {"adamw", "muon", "normuon"}:
      raise ValueError(f"unsupported optimizer.kind: {self.optimizer.kind}")
    return TrainConfig(
      data=self.data,
      model=self.model,
      optimizer=self.optimizer,
      schedule=self.schedule,
      loop=self.loop,
      eval=self.eval,
      rollout=self.rollout,
      checkpoint=self.checkpoint,
    )


@dataclass(frozen=True)
class ExperimentOverrides:
  device: str | None = None
  steps: int | None = None
  root_dir: Path | None = None
  seed: int | None = None


@dataclass(frozen=True)
class WorkspaceOverrides:
  code_root: Path | None = None
  workspace_root: Path | None = None
  runs_dir: Path | None = None
  data_dir: Path | None = None
  tokenizers_dir: Path | None = None
  models_dir: Path | None = None
  cache_dir: Path | None = None


def load_experiment_config(path: Path) -> ExperimentConfig:
  raw = yaml.safe_load(path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError("experiment config must be a YAML mapping")
  return parse_experiment_config(raw)


def load_workspace_config(path: Path | None = None) -> WorkspaceConfig:
  if path is None:
    path = discover_workspace_config()
  if path is None:
    return WorkspaceConfig()
  raw = yaml.safe_load(path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError("workspace config must be a YAML mapping")
  return parse_workspace_config(raw)


def discover_workspace_config(start: Path | None = None) -> Path | None:
  current = (start or Path.cwd()).resolve()
  if current.is_file():
    current = current.parent
  for directory in (current, *current.parents):
    path = directory / WORKSPACE_CONFIG_NAME
    if path.is_file():
      return path
  return None


def parse_workspace_config(raw: dict[str, Any]) -> WorkspaceConfig:
  allowed = {"dirs", "workspace"}
  unknown = set(raw) - allowed
  if unknown:
    raise ValueError(f"unknown workspace config keys: {sorted(unknown)}")
  dirs = _section(raw, "dirs")
  workspace = _section(raw, "workspace")
  _reject_unknown(dirs, {"code_root", "workspace_root"}, "workspace dirs")
  _reject_unknown(
    workspace,
    {
      "runs_dir",
      "data_dir",
      "tokenizers_dir",
      "models_dir",
      "cache_dir",
    },
    "workspace",
  )
  if "code_root" not in dirs or "workspace_root" not in dirs:
    raise ValueError("workspace dirs must include code_root and workspace_root")
  return WorkspaceConfig(
    code_root=Path(dirs["code_root"]),
    workspace_root=Path(dirs["workspace_root"]),
    runs_dir=Path(workspace.get("runs_dir", "runs")),
    data_dir=Path(workspace.get("data_dir", "data")),
    tokenizers_dir=Path(workspace.get("tokenizers_dir", "tokenizers")),
    models_dir=Path(workspace.get("models_dir", "models")),
    cache_dir=Path(workspace.get("cache_dir", "cache")),
  )


def parse_experiment_config(raw: dict[str, Any]) -> ExperimentConfig:
  allowed = {
    "name",
    "data",
    "model",
    "optimizer",
    "schedule",
    "loop",
    "eval",
    "rollout",
    "checkpoint",
    "system_metrics",
    "run",
    "secrets",
    "output",
    "sinks",
  }
  unknown = set(raw) - allowed
  if unknown:
    raise ValueError(f"unknown experiment config keys: {sorted(unknown)}")
  if "name" not in raw:
    raise ValueError("experiment config requires 'name'")

  data = _section(raw, "data")
  model = _section(raw, "model")
  optimizer = _section(raw, "optimizer")
  schedule = _section(raw, "schedule")
  loop = _section(raw, "loop")
  eval_config = _optional_section(raw, "eval")
  rollout = _optional_section(raw, "rollout")
  checkpoint = _section(raw, "checkpoint")
  system_metrics = _section(raw, "system_metrics")
  run = _section(raw, "run")
  secrets = _section(raw, "secrets")
  output = _section(raw, "output")

  return ExperimentConfig(
    name=str(raw["name"]),
    data=_parse_data(data),
    model=_parse_model(model),
    optimizer=OptimizerConfig(
      kind=optimizer.get("kind", "adamw"),
      learning_rate=float(optimizer.get("learning_rate", 3e-4)),
      weight_decay=float(optimizer.get("weight_decay", 0.1)),
      max_grad_norm=_optional_float(optimizer.get("max_grad_norm", 1.0)),
    ),
    schedule=OptimizerScheduleConfig(
      warmup_steps=int(schedule.get("warmup_steps", 0)),
      cooldown_steps=int(schedule.get("cooldown_steps", 0)),
      final_lr_scale=float(schedule.get("final_lr_scale", 0.0)),
      momentum_start=_optional_float(schedule.get("momentum_start")),
      momentum_end=_optional_float(schedule.get("momentum_end")),
      momentum_warmup_steps=int(schedule.get("momentum_warmup_steps", 0)),
      scale_weight_decay_with_lr=bool(
        schedule.get("scale_weight_decay_with_lr", False)
      ),
    ),
    loop=LoopConfig(
      seed=int(loop.get("seed", 42)),
      device=str(loop.get("device", "cpu")),
      dtype=_parse_torch_dtype(loop.get("dtype", "float32")),
      batch_size=_parse_batch_size(loop.get("batch_size", 8)),
      batch_size_vram_fraction=float(loop.get("batch_size_vram_fraction", 0.9)),
      steps=int(loop.get("steps", 10)),
    ),
    eval=_parse_eval(eval_config),
    rollout=_parse_rollout(rollout),
    checkpoint=_parse_checkpoint(checkpoint),
    system_metrics=_parse_system_metrics(system_metrics),
    run=_parse_run(run),
    secrets=SecretsConfig(
      env_file=_optional_path(secrets.get("env_file", ".secret")),
    ),
    output=_parse_output(output),
    sinks=_parse_sinks(raw.get("sinks")),
  )


def apply_overrides(
  config: ExperimentConfig,
  overrides: ExperimentOverrides,
) -> ExperimentConfig:
  return ExperimentConfig(
    name=config.name,
    data=config.data,
    model=config.model,
    optimizer=config.optimizer,
    schedule=config.schedule,
    loop=LoopConfig(
      seed=config.loop.seed if overrides.seed is None else overrides.seed,
      device=config.loop.device if overrides.device is None else overrides.device,
      dtype=config.loop.dtype,
      batch_size=config.loop.batch_size,
      batch_size_vram_fraction=config.loop.batch_size_vram_fraction,
      steps=config.loop.steps if overrides.steps is None else overrides.steps,
    ),
    eval=config.eval,
    rollout=config.rollout,
    checkpoint=config.checkpoint,
    system_metrics=config.system_metrics,
    run=config.run,
    secrets=config.secrets,
    output=config.output
    if overrides.root_dir is None
    else OutputConfig(root_dir=overrides.root_dir),
    sinks=config.sinks,
  )


def apply_workspace_overrides(
  config: WorkspaceConfig,
  overrides: WorkspaceOverrides,
) -> WorkspaceConfig:
  return WorkspaceConfig(
    code_root=config.code_root if overrides.code_root is None else overrides.code_root,
    workspace_root=config.workspace_root
    if overrides.workspace_root is None
    else overrides.workspace_root,
    runs_dir=config.runs_dir if overrides.runs_dir is None else overrides.runs_dir,
    data_dir=config.data_dir if overrides.data_dir is None else overrides.data_dir,
    tokenizers_dir=config.tokenizers_dir
    if overrides.tokenizers_dir is None
    else overrides.tokenizers_dir,
    models_dir=config.models_dir
    if overrides.models_dir is None
    else overrides.models_dir,
    cache_dir=config.cache_dir if overrides.cache_dir is None else overrides.cache_dir,
  )


def resolve_workspace_paths(
  config: ExperimentConfig,
  workspace: WorkspaceConfig,
) -> ExperimentConfig:
  workspace_root = workspace.workspace_root
  return ExperimentConfig(
    name=config.name,
    data=replace(
      config.data,
      dataset_root=_resolve_against(workspace_root, config.data.dataset_root),
      encoding_name=_resolve_encoding_name(workspace_root, config.data.encoding_name),
    ),
    model=config.model,
    optimizer=config.optimizer,
    schedule=config.schedule,
    loop=config.loop,
    eval=config.eval,
    rollout=config.rollout,
    checkpoint=replace(
      config.checkpoint,
      resume=_resolve_checkpoint_resume(workspace_root, config.checkpoint.resume),
    ),
    system_metrics=config.system_metrics,
    run=config.run,
    secrets=SecretsConfig(
      env_file=None
      if config.secrets.env_file is None
      else _resolve_against(workspace_root, config.secrets.env_file),
    ),
    output=OutputConfig(root_dir=workspace.runs_path),
    sinks=tuple(_resolve_sink_paths(workspace_root, sink) for sink in config.sinks),
  )


def config_to_plain(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, tuple):
    return [config_to_plain(item) for item in value]
  if hasattr(value, "__dataclass_fields__"):
    return {key: config_to_plain(item) for key, item in asdict(value).items()}
  if isinstance(value, dict):
    return {key: config_to_plain(item) for key, item in value.items()}
  if isinstance(value, list):
    return [config_to_plain(item) for item in value]
  return value


def write_yaml(path: Path, value: Any) -> None:
  path.write_text(
    yaml.safe_dump(value, sort_keys=False),
    encoding="utf-8",
  )


def _reject_unknown(
  value: dict[str, Any],
  allowed: set[str],
  section: str,
) -> None:
  unknown = set(value) - allowed
  if unknown:
    raise ValueError(f"unknown {section} config keys: {sorted(unknown)}")


def _resolve_against(base: Path, path: Path) -> Path:
  if path.is_absolute():
    return path
  return base / path


def _resolve_encoding_name(base: Path, encoding_name: str) -> str:
  for prefix in ("unitoken:", "repo_bpe:"):
    if encoding_name.startswith(prefix):
      path = Path(encoding_name.removeprefix(prefix))
      return f"{prefix}{_resolve_against(base, path).as_posix()}"
  repo_bpe_backend_prefix = "repo_bpe+"
  if encoding_name.startswith(repo_bpe_backend_prefix):
    backend, separator, path = encoding_name.removeprefix(
      repo_bpe_backend_prefix
    ).partition(":")
    if separator == ":" and path:
      resolved = _resolve_against(base, Path(path)).as_posix()
      return f"{repo_bpe_backend_prefix}{backend}:{resolved}"
  return encoding_name


def _resolve_checkpoint_resume(base: Path, resume: str | None) -> str | None:
  if resume is None or resume == "auto":
    return resume
  return str(_resolve_against(base, Path(resume)))


def _resolve_sink_paths(base: Path, sink: SinkConfig) -> SinkConfig:
  if isinstance(sink, FilesSinkConfig):
    return replace(
      sink,
      run_dir=None if sink.run_dir is None else _resolve_against(base, sink.run_dir),
    )
  if isinstance(sink, TensorBoardSinkConfig):
    return replace(
      sink,
      log_dir=None if sink.log_dir is None else _resolve_against(base, sink.log_dir),
    )
  if isinstance(sink, WandbSinkConfig):
    return replace(
      sink,
      dir=None if sink.dir is None else _resolve_against(base, sink.dir),
    )
  return sink


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
  value = raw.get(name, {})
  if not isinstance(value, dict):
    raise ValueError(f"experiment config section '{name}' must be a mapping")
  return value


def _optional_section(raw: dict[str, Any], name: str) -> dict[str, Any] | None:
  if name not in raw:
    return None
  if raw[name] is None:
    return None
  return _section(raw, name)


def _optional_int(value: Any) -> int | None:
  if value is None:
    return None
  return int(value)


def _optional_float(value: Any) -> float | None:
  if value is None:
    return None
  return float(value)


def _optional_path(value: Any) -> Path | None:
  if value is None:
    return None
  return Path(value)


def _parse_data(value: dict[str, Any]) -> DataConfig:
  allowed = {
    "kind",
    "encoding_name",
    "seq_len",
    "dataset_root",
    "version",
    "split",
    "resolved_version",
  }
  unknown = set(value) - allowed
  if unknown:
    raise ValueError(f"unknown data config keys: {sorted(unknown)}")
  kind = value.get("kind", "token_dataset")
  if kind != "token_dataset":
    raise ValueError(f"unsupported data.kind: {kind}")
  split = str(value.get("split", "train"))
  if split not in {"train", "val", "test"}:
    raise ValueError(f"unsupported data.split: {split}")
  return DataConfig(
    kind=kind,
    encoding_name=str(value.get("encoding_name", "cl100k_base")),
    seq_len=int(value.get("seq_len", 128)),
    dataset_root=Path(value.get("dataset_root", "cache/repo_sources_cl100k")),
    version=str(value.get("version", "latest")),
    split=split,
    resolved_version=value.get("resolved_version"),
  )


def _parse_torch_dtype(value: Any) -> TorchDType:
  dtype = str(value)
  aliases = {
    "fp32": "float32",
    "float": "float32",
    "fp16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
  }
  dtype = aliases.get(dtype, dtype)
  if dtype not in {"float32", "float16", "bfloat16"}:
    raise ValueError(f"unsupported loop.dtype: {dtype}")
  return dtype


def _parse_batch_size(value: Any) -> int | Literal["auto"]:
  if str(value) == "auto":
    return "auto"
  batch_size = int(value)
  if batch_size < 1:
    raise ValueError("loop.batch_size must be positive or 'auto'")
  return batch_size


def _parse_eval(value: dict[str, Any] | None) -> EvalConfig | None:
  if value is None:
    return None
  allowed = {"split", "every_steps", "max_batches"}
  unknown = set(value) - allowed
  if unknown:
    raise ValueError(f"unknown eval config keys: {sorted(unknown)}")
  split = str(value.get("split", "test"))
  if split not in {"val", "test"}:
    raise ValueError(f"unsupported eval.split: {split}")
  return EvalConfig(
    split=split,
    every_steps=int(value.get("every_steps", 100)),
    max_batches=int(value.get("max_batches", 8)),
  )


def _parse_rollout(value: dict[str, Any] | None) -> RolloutConfig | None:
  if value is None:
    return None
  allowed = {"every_steps", "max_new_tokens", "prompts"}
  unknown = set(value) - allowed
  if unknown:
    raise ValueError(f"unknown rollout config keys: {sorted(unknown)}")
  return RolloutConfig(
    every_steps=int(value.get("every_steps", 100)),
    max_new_tokens=int(value.get("max_new_tokens", 64)),
    prompts=_parse_rollout_prompts(value.get("prompts")),
  )


def _parse_checkpoint(value: dict[str, Any]) -> CheckpointConfig:
  allowed = {"enabled", "every_steps", "keep_last", "resume"}
  unknown = set(value) - allowed
  if unknown:
    raise ValueError(f"unknown checkpoint config keys: {sorted(unknown)}")
  every_steps = int(value.get("every_steps", 100))
  keep_last = int(value.get("keep_last", 3))
  if every_steps <= 0:
    raise ValueError("checkpoint.every_steps must be positive")
  if keep_last < 0:
    raise ValueError("checkpoint.keep_last must be non-negative")
  return CheckpointConfig(
    enabled=bool(value.get("enabled", False)),
    every_steps=every_steps,
    keep_last=keep_last,
    resume=_optional_str(value.get("resume")),
  )


def _parse_system_metrics(value: dict[str, Any]) -> SystemMetricsConfig:
  allowed = {"enabled", "every_seconds"}
  unknown = set(value) - allowed
  if unknown:
    raise ValueError(f"unknown system_metrics config keys: {sorted(unknown)}")
  every_seconds = float(value.get("every_seconds", 5.0))
  if every_seconds <= 0:
    raise ValueError("system_metrics.every_seconds must be positive")
  return SystemMetricsConfig(
    enabled=bool(value.get("enabled", True)),
    every_seconds=every_seconds,
  )


def _parse_run(value: dict[str, Any]) -> RunConfig:
  allowed = {"id", "name", "group"}
  unknown = set(value) - allowed
  if unknown:
    raise ValueError(f"unknown run config keys: {sorted(unknown)}")
  return RunConfig(
    id=_optional_str(value.get("id")),
    name=_optional_str(value.get("name")),
    group=_optional_str(value.get("group")),
  )


def _parse_output(value: dict[str, Any]) -> OutputConfig:
  allowed = {"root_dir"}
  unknown = set(value) - allowed
  if unknown:
    raise ValueError(f"unknown output config keys: {sorted(unknown)}")
  return OutputConfig(
    root_dir=Path(value.get("root_dir", "runs")),
  )


def _parse_rollout_prompts(value: Any) -> tuple[RolloutPromptConfig, ...]:
  if value is None:
    return ()
  if not isinstance(value, list | tuple):
    raise ValueError("rollout.prompts must be a list")
  prompts = []
  for item in value:
    if not isinstance(item, dict):
      raise ValueError("rollout prompt must be a mapping")
    prompts.append(
      RolloutPromptConfig(
        name=str(item.get("name", "prompt")),
        prompt=str(item.get("prompt", "")),
      )
    )
  return tuple(prompts)


def _parse_model(value: dict[str, Any]) -> ModelConfig:
  kind = value.get("kind", "reference")
  attention_backend = str(value.get("attention_backend", "torch"))
  if attention_backend not in {"torch", "flash_attention2", "tilelang"}:
    raise ValueError(f"unsupported model.attention_backend: {attention_backend}")
  loss_backend = str(value.get("loss_backend", "cross_entropy"))
  if loss_backend not in {
    "cross_entropy",
    "linear_cross_entropy",
    "cut_cross_entropy",
    "tilelang_linear_cross_entropy",
  }:
    raise ValueError(f"unsupported model.loss_backend: {loss_backend}")
  loss_chunk_size = int(value.get("loss_chunk_size", 512))
  if loss_chunk_size <= 0:
    raise ValueError("model.loss_chunk_size must be positive")
  if kind == "reference":
    return ReferenceModelConfig(
      d_model=int(value.get("d_model", 128)),
      n_layers=int(value.get("n_layers", 2)),
      n_heads=int(value.get("n_heads", 4)),
      d_ff=_optional_int(value.get("d_ff")),
      attention_backend=attention_backend,
      loss_backend=loss_backend,
      loss_chunk_size=loss_chunk_size,
    )
  if kind == "nanogpt_speedrun":
    return NanoGPTSpeedrunModelConfig(
      d_model=int(value.get("d_model", 768)),
      n_layers=int(value.get("n_layers", 11)),
      n_heads=int(value.get("n_heads", 12)),
      d_ff=int(value.get("d_ff", 3072)),
      attention_backend=attention_backend,
      loss_backend=loss_backend,
      loss_chunk_size=loss_chunk_size,
      logit_softcap=_optional_float(value.get("logit_softcap", 30.0)),
      logit_scale=float(value.get("logit_scale", 1.0)),
      logit_sigmoid_scale=_optional_float(value.get("logit_sigmoid_scale", 23.0)),
      logit_sigmoid_bias=float(value.get("logit_sigmoid_bias", 5.0)),
      logit_sigmoid_temperature=float(value.get("logit_sigmoid_temperature", 7.5)),
      token_smear=bool(value.get("token_smear", True)),
      smear_gate_dim=int(value.get("smear_gate_dim", 12)),
      partial_key_offset_layers=_int_tuple(
        value.get("partial_key_offset_layers", (3, 10))
      ),
      attention_gate_dim=int(value.get("attention_gate_dim", 12)),
      xsa=bool(value.get("xsa", True)),
      attention_free_layer=_optional_int(value.get("attention_free_layer", 6)),
      bigram_vocab_size=_optional_int(value.get("bigram_vocab_size")),
      bigram_dim=int(value.get("bigram_dim", 192)),
      bigram_sign_table_rows=int(value.get("bigram_sign_table_rows", 8192)),
      mtp_weights=_float_tuple(value.get("mtp_weights", (1.0, 0.5, 0.25))),
      embedding_skip=bool(value.get("embedding_skip", True)),
      value_residual=bool(value.get("value_residual", True)),
      block_skip_from=_optional_int(value.get("block_skip_from", 3)),
      block_skip_to=_optional_int(value.get("block_skip_to", 6)),
      residual_decay=float(value.get("residual_decay", 1.0)),
      tie_embeddings=bool(value.get("tie_embeddings", True)),
    )
  if kind == "ds_tiny":
    return DSTinyModelConfig(
      d_model=int(value.get("d_model", 128)),
      n_layers=int(value.get("n_layers", 2)),
      n_heads=int(value.get("n_heads", 4)),
      d_ff=_optional_int(value.get("d_ff")),
      q_lora_rank=_optional_int(value.get("q_lora_rank")),
      kv_lora_rank=int(value.get("kv_lora_rank", 64)),
      qk_nope_head_dim=int(value.get("qk_nope_head_dim", 16)),
      qk_rope_head_dim=int(value.get("qk_rope_head_dim", 16)),
      v_head_dim=int(value.get("v_head_dim", 32)),
      attention_backend=attention_backend,
      loss_backend=loss_backend,
      loss_chunk_size=loss_chunk_size,
    )
  if kind == "deepseek_v4":
    return DeepSeekV4ModelConfig(
      d_model=int(value.get("d_model", 128)),
      n_layers=int(value.get("n_layers", 2)),
      n_heads=int(value.get("n_heads", 4)),
      head_dim=_optional_int(value.get("head_dim")),
      d_ff=_optional_int(value.get("d_ff")),
      q_lora_rank=_optional_int(value.get("q_lora_rank")),
      kv_lora_rank=int(value.get("kv_lora_rank", 64)),
      qk_nope_head_dim=int(value.get("qk_nope_head_dim", 16)),
      qk_rope_head_dim=int(value.get("qk_rope_head_dim", 16)),
      v_head_dim=int(value.get("v_head_dim", 32)),
      rope_head_dim=_optional_int(value.get("rope_head_dim")),
      o_lora_rank=_optional_int(value.get("o_lora_rank")),
      o_groups=int(value.get("o_groups", 1)),
      attention_layer_types=_optional_str_tuple(value.get("attention_layer_types")),
      compress_rate_csa=int(value.get("compress_rate_csa", 4)),
      compress_rate_hca=int(value.get("compress_rate_hca", 128)),
      index_n_heads=int(value.get("index_n_heads", 64)),
      index_head_dim=int(value.get("index_head_dim", 128)),
      index_topk=int(value.get("index_topk", 512)),
      n_routed_experts=int(value.get("n_routed_experts", 4)),
      n_shared_experts=int(value.get("n_shared_experts", 1)),
      n_experts_per_token=int(value.get("n_experts_per_token", 2)),
      n_group=int(value.get("n_group", 2)),
      topk_group=int(value.get("topk_group", 1)),
      dense_layers=int(value.get("dense_layers", 1)),
      attention_backend=attention_backend,
      loss_backend=loss_backend,
      loss_chunk_size=loss_chunk_size,
    )
  raise ValueError(f"unsupported model.kind: {kind}")


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
  if value is None:
    return None
  if not isinstance(value, list | tuple):
    raise ValueError("attention_layer_types must be a list")
  return tuple(str(item) for item in value)


def _int_tuple(value: Any) -> tuple[int, ...]:
  if not isinstance(value, list | tuple):
    raise ValueError("expected a list of integers")
  return tuple(int(item) for item in value)


def _float_tuple(value: Any) -> tuple[float, ...]:
  if not isinstance(value, list | tuple):
    raise ValueError("expected a list of numbers")
  return tuple(float(item) for item in value)


def _optional_float(value: Any) -> float | None:
  if value is None:
    return None
  return float(value)


def _optional_str(value: Any) -> str | None:
  if value is None:
    return None
  return str(value)


def _parse_sinks(value: Any) -> tuple[SinkConfig, ...]:
  if value is None:
    return ()
  if not isinstance(value, list | tuple):
    raise ValueError("sinks must be a list")
  return tuple(_parse_sink(item) for item in value)


def _parse_sink(value: Any) -> SinkConfig:
  if not isinstance(value, dict):
    raise ValueError("sink config must be a mapping")
  kind = value.get("kind")
  if kind == "files":
    return FilesSinkConfig(
      run_dir=_optional_path(value.get("run_dir")),
      config_json=str(value.get("config_json", "config.json")),
      resolved_config_yaml=str(
        value.get("resolved_config_yaml", "config.resolved.yaml")
      ),
      status_json=str(value.get("status_json", "status.json")),
      metrics_jsonl=str(value.get("metrics_jsonl", "metrics.jsonl")),
      system_metrics_jsonl=str(
        value.get("system_metrics_jsonl", "system_metrics.jsonl")
      ),
      result_json=str(value.get("result_json", "result.json")),
    )
  if kind == "tensorboard":
    return TensorBoardSinkConfig(
      log_dir=_optional_path(value.get("log_dir")),
      flush_secs=int(value.get("flush_secs", 10)),
    )
  if kind == "mlflow":
    return MlflowSinkConfig(
      tracking_uri=value.get("tracking_uri"),
      experiment_name=_optional_str(value.get("experiment_name")),
      run_name=value.get("run_name"),
      nested=bool(value.get("nested", False)),
    )
  if kind == "wandb":
    return WandbSinkConfig(
      project=str(value.get("project", "flm")),
      entity=value.get("entity"),
      name=value.get("name"),
      mode=value.get("mode"),
      dir=_optional_path(value.get("dir")),
      tags=_str_tuple(value.get("tags")),
      group=value.get("group"),
      job_type=value.get("job_type"),
    )
  raise ValueError(f"unsupported sink kind: {kind}")


def _str_tuple(value: Any) -> tuple[str, ...]:
  if value is None:
    return ()
  if not isinstance(value, list | tuple):
    raise ValueError("tags must be a list")
  return tuple(str(item) for item in value)
