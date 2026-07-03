# Glossary

Chinese translations under `blog/cn/` keep recurring technical entity names
in English, so readers can map prose directly back to the code. This file is
the authoritative list of those terms.

## Terms kept in English

| English        | Avoid (Chinese)   | Notes                                                       |
| -------------- | ----------------- | ----------------------------------------------------------- |
| Model          | 模型               | Model architecture / instance                               |
| Dataset        | 数据集             | Data loading, dataset support                               |
| Optimizer      | 优化器             | AdamW, optimizer config                                     |
| Protocol       | 协议               | Python `Protocol`, e.g. `RunSink`, `LanguageModel`          |
| Registry       | 注册表             | Sink registry                                               |
| Config         | 配置               | Experiment / model / training config                        |
| loss           | 损失               | training loss, loss curve                                   |
| sink           | （不译）           | Metric sink backend                                         |
| run / runner   | （不译）           | A training run, the `ExperimentRunner`                      |
| trainer        | （不译）           | The reusable training engine                                |

## Always kept in English (code identifiers)

These are never translated — they are literal code: package and module names
(`flm-llm`, `flm-train`, `sinks/`, `trainer.py`, `config.py`), class and
type names (`ReferenceModel`, `DSTiny`, `DeepSeekV4`, `RunSink`,
`CompositeRunSink`, `ExperimentConfig`, `TrainStepMetrics`, `LanguageModel`),
backend names (`files`, `TensorBoard`, `MLflow`, `Weights & Biases` / W&B),
and all CLI commands and flags.

## Keep in Chinese

Section chrome and general prose stay Chinese: 标题, 目标, 任务看板, 如何
使用, 起点, 小结, 开闭原则, 训练循环 (loop), 引擎 (engine), 后端 (backend),
密钥 (secret), etc. The English-keep rule applies to the recurring technical
*nouns* above, not to ordinary words.

## Numbers

Use the compact form **16M** for parameter counts, not 万 (e.g. "16M 参数",
not "1600 万参数").
