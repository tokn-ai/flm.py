# 从脚本到系统：重塑训练工作流的一天

这是记录 **FLM** 构建过程的第一篇日志。FLM 是一个用于 LLM 实验的轻量 `uv` 单仓库。

## 目标

> 端到端训练一个真正可用的 **16M 参数**语言模型，并保证整个工作流可复现、可观测、易扩展。

### 任务看板

- [x] `flm-llm` 中的 Model 架构：ReferenceModel、DSTiny、DeepSeekV4
- [x] `flm-modules` 中的神经网络基础组件 + AdamW Optimizer
- [x] `flm-datasets` 中的 Dataset 加载（含 CalcQA）
- [x] `flm-rl` 中的强化学习训练器（PPO、GRPO）
- [ ] (current) Config 驱动的训练：YAML 实验 Config、模块拆分、按类型区分的 Config、本地密钥加载
- [ ] (current) 可复用的训练引擎与可插拔的 sink（files、TensorBoard、MLflow、W&B）
- [ ] 评测流程（实际基准指标，而不只是训练 loss）
- [ ] 训练中存档 + 断点续训
- [ ] 在仓库源预设之外，支持更大规模的真实 Dataset

---

今天的提交与模型质量无关。它们的目标是把一个一次性的训练*脚本*变成一个*系统*——Config 驱动、可插拔、低耦合。本文是这次变化的高层概览，以及背后的原因。

## 如何使用

一次运行就是一个 Config 文件加上一条命令。仓库随附了 `experiments/16m_repo.yaml` 作为起点：

```sh
uv run flm-train-experiment experiments/16m_repo.yaml
```

快速跑一个只走一步就退出的 CPU 冒烟测试：

```sh
uv run flm-train-experiment experiments/16m_repo.yaml \
  --device cpu \
  --steps 1 \
  --run-dir /tmp/flm-experiment-smoke
```

其余的一切——Model、数据、Optimizer、循环、sink——都在 YAML 里声明，不在命令行上。

## 起点

今天之前，训练逻辑挤在两个大文件里：一个是仓库专用的 `train.py`，把 CLI 参数解析、Model 构建、训练循环和指标写入全部塞进了一个 260 行的脚本；另一个是 `experiment.py`，在另一层做着几乎同样的事。它能跑，但很难扩展：每加一个新 Model、新 Dataset 或新日志后端，都得改同一个纠缠不清的文件；而且运行靠命令行参数驱动，而不是被记录下来的 Config。

对于一个朝向真正 16M 训练的项目来说，这个形状是错的。于是今天用来重塑它。

## 变了什么

这一天的主线是**关注点分离**。工作可以拆成四条线索。

### 1. Config 作为唯一真相源

现在运行由 `experiments/` 下的 YAML 文件驱动，从 `16m_repo.yaml` 开始。一次运行不再需要一长串命令行参数——你把 runner 指向一个 Config，它会把一切都解析好：

```sh
uv run flm-train-experiment experiments/16m_repo.yaml
```

旧的 `experiment.py` 大块头被拆成职责清晰的模块：`config.py`（一整套 frozen dataclass + YAML 加载）、`cli.py`（薄薄一层参数解析）、`runner.py`（编排）。Config 是数据，不是代码。

### 2. 一个可复用的训练引擎

真正的逐步训练循环被从实验代码里提取出来，放进 `trainer.py`——一个通用引擎，由一个轻量的 `LanguageModel` Protocol 驱动，并产出结构化的 `TrainStepMetrics`。实验 runner *使用*训练器，而不是*包含*训练器。旧的仓库专用 `train.py` 被整个删掉，取而代之的是薄薄的 `data.py` / `models.py` / `presets.py` 辅助层。训练循环现在只有一份实现。

### 3. 可插拔的指标 sink

可观测性从一个单一的 `sinks.py` 文件，升级成一个基于 `RunSink` Protocol 和 Registry 的 `sinks/` 包。这个 Protocol 刻意做得很小——一个后端只需要实现这几个钩子：

```python
class RunSink(Protocol):
  def start_run(self, context: RunContext, config: ExperimentConfig) -> None: ...
  def write_config(self, config: ExperimentConfig) -> None: ...
  def log_status(self, status: RunStatus, message: str | None = None) -> None: ...
  def log_metrics(self, metrics: dict[str, Scalar], step: int) -> None: ...
  def log_artifact(self, path: Path, name: str | None = None) -> None: ...
  def finish_run(self, result: TrainingResult) -> None: ...
  def close(self) -> None: ...
```

今天交付了四个真正的后端——`files`（磁盘上的 JSON/JSONL 产物）、`TensorBoard`、`MLflow` 和 `Weights & Biases`——而一个 YAML Config 只需列出一次运行要用哪几个。加一个新后端意味着实现该 Protocol 并注册它；别的什么都不用改。甚至还有一个 `CompositeRunSink`，会把每一次调用扇出到一组 sink，于是一次运行可以同时写磁盘并推送到 W&B。这就是开闭原则，用最朴素的方式挣来的。

### 4. 能随项目一起成长的 Config

Model Config 被重新组织成**按类型**的辨识联合（`ReferenceModelConfig`、`DSTinyModelConfig`、`DeepSeekV4ModelConfig`），每一个都带一个 `Literal` 标签，这样随着架构增多，YAML 也始终没有歧义。训练 Config 被归并成连贯的几块（`TrainConfig`、`LoopConfig`、`OptimizerConfig`、`DataConfig`）。还有一个小小的 `secrets.py`，负责加载本地的 `.secret` 环境文件，于是像 HuggingFace 或 W&B 的密钥永远不会泄露进仓库或命令行。

## 为什么这很重要

今天的提交没有任何一条会改变 loss 曲线。它们改变的是下一次改动的成本。当评测接下来落地时，它会接进一个 Config 和一个 sink——而不是一次重写。当存档功能到来时，它会进到一个早已把编排与训练循环分开的 runner 里。16M 的训练是目的地；今天做的是把通往它的路铺好。

## 小结

在一天之内，`flm-train` 从一个"恰好能训练模型"的脚本，变成了一个小系统：Config 驱动、引擎可复用、sink 可插拔、密钥安全。模型本身原封未动。这正是重点——今天的工作，是为了让*下一个* Model 更容易上线。
