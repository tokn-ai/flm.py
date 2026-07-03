# 干掉 logits 张量：为 16M 训练引入 cut cross-entropy

FLM 构建日志的第二篇。昨天讲的是*塑造*训练工作流；今天只聊一个数字：**loss 占多少 GPU 显存**，以及为什么最直观的算法要花 ~6 GiB，而实际只需 ~25 MiB。

## 目标

> 端到端训练一个真正可用的 **16M 参数**语言模型，并保证整个工作流可复现、可观测、易扩展。

### 任务看板

- [x] `flm-llm` 中的 Model 架构：ReferenceModel、DSTiny、DeepSeekV4
- [x] `flm-modules` 中的神经网络基础组件 + AdamW Optimizer
- [x] `flm-datasets` 中的 Dataset 加载（含 CalcQA）
- [x] `flm-rl` 中的强化学习训练器（PPO、GRPO）
- [x] Config 驱动的训练：YAML Config、模块拆分、按类型区分的 Config、本地密钥加载
- [x] 可复用的训练引擎与可插拔的 sink（files、TensorBoard、MLflow、W&B）
- [ ] (current) 显存高效的 loss backend：torch linear cross-entropy、TileLang CCE、Cut Cross-Entropy
- [ ] 评测流程（实际基准指标，而不只是训练 loss）
- [ ] 训练中存档 + 断点续训
- [ ] 在仓库源预设之外，支持更大规模的真实 Dataset

---

## 如何使用

loss backend 是 Model 上的一个 YAML 旋钮，计算 dtype 是 loop 上的一个旋钮。切到融合、显存高效的 loss 只需改一行：

```yaml
model:
  kind: reference
  loss_backend: cut_cross_entropy   # 或 linear_cross_entropy / tilelang_linear_cross_entropy
  loss_chunk_size: 512

loop:
  dtype: bfloat16
```

无需改代码——trainer 通过 `flm-modules` 里的 `language_model_loss(...)` [5] 分发，Model 永远不会构建完整的 logits 矩阵。

## 罪魁祸首：一张 10 万宽的 logits 张量

我们的 tokenizer 是 `cl100k_base`，所以词表大小是 **V ≈ 100,000**。语言建模头把每个 token 的 hidden state 映射到整张词表上的分布。经典做法分两步：

```python
logits = F.linear(hidden, classifier_weight)   # [N, V]
loss = F.cross_entropy(logits, targets)
```

其中 `N` 是这一步的 token 数，`V ≈ 100k`。问题出在中间产物 `logits`：它是 `N × V` 个元素，而 `V` 极大。在一个 `N = 4096` 个 token 的训练 step 里，那就是约 4 亿个元素——而在反向传播期间，正好是这个形状的若干张量同时存活。

## 数学上：每个 token 只需两个数

设 $h_n \in \mathbb{R}^d$ 为 token $n$ 的 hidden state，$w_v \in
\mathbb{R}^d$ 为词表第 $v$ 项的 classifier weight，$y_n$ 为目标索引。两步法是：

$$
\text{logits}_{n,v} = h_n^\top w_v
$$

$$
L = \frac{1}{N}\sum_n \text{CE}(\text{softmax}(\text{logits}_n),\, y_n)
  = \frac{1}{N}\sum_n \left[ -\text{logits}_{n,y_n} + \log\sum_v \exp(\text{logits}_{n,v}) \right]
$$

第一项只是与单个目标列的点积；第二项是对词表的 `logsumexp`——一个归约。所以对每个 token，loss 只需要从 `[N, V]` 的 logits 里导出两个标量，于是整个式子可以折叠成：

$$
L = \frac{1}{N}\sum_n \left[ \operatorname{logsumexp}_v(h_n^\top w_v) - h_n^\top w_{y_n} \right]
$$

这两项都不需要整张矩阵同时存在。我们可以按列块逐块计算点积，把每一块折进一个运行的 `logsumexp` [2]，然后丢掉——只保留一小片 `[chunk, V]` 加上每个 token 一个累加器。这正是 CCE 所做的融合。

## 为什么旧路径吃掉 ~6 GiB

在训练规模下——batch 8 × seq 512——一步带的是 **N = 4096 个 token**，词表 **V ≈ 100,000**。那么一张 fp32 的 `[N, V]` 张量就是：

```
4096 × 100,000 × 4 B ≈ 1.5 GiB
```

`F.cross_entropy` 为了数值稳定性，会在计算 log-softmax 之前**把输入从 bf16 上转（upcast）到 fp32**，于是 loss 区域即便网络其余部分都在 bf16 下运行，也按每元素 4 字节来算。走一遍 `F.linear` + `F.cross_entropy` 的 autograd 图，你会发现反向期间大约有四个完整的 `[N, V]` 张量常驻：

- 被保存的 `logits`（bf16），
- fp32 上转副本，
- softmax 概率（fp32），
- `grad_logits`（`softmax − onehot`，fp32）。

四个块每个约 1.5 GiB：

```
4 × 1.5 GiB ≈ 6 GiB
```

这就是单独 loss 区域的反向峰值：**~6 GiB**，几乎全花在了一些张量上——而它们存在的唯一目的，就是紧接着被归约成每个 token 一个标量。它随 `N × V` 线性增长：batch 翻倍或词表变大，每个块就再添约 1.5 GiB。在一块瞄准 16M 训练的小 GPU 上，这就是能跑和 OOM 之间的差别。

## 为什么 CCE 把 loss 区域降到 ~25 MiB

办法是**永远不要把 logits 物化出来**。Cross-entropy 本质是个归约：对每个 token，我们只需要整张词表上的 logsumexp，以及目标 token 的那条 logit。Cut Cross-Entropy（CCE）把 matmul 和 loss 融合成一个 kernel，按块在词表维度上流式处理，累加一个很小的、每个 token 一份的 `logsumexp`，而不是整张 `[N, V]` 矩阵。

loss 区域里仍然常驻的是：

- `hidden` `[N, d]`——很小（`d` 只有几百）。
- `classifier_weight` `[V, d]`——绑定的 embedding，本来就一直常驻，不算额外开销。
- 每个 token 一份的 `logsumexp` `[N]`——单个向量，可忽略。
- 一小块工作中的 logits `[chunk, V]`，fp32——唯一真正的分配。

以 64 行的 chunk 为例，这一小片是：

```
64 × 100,000 × 4 B ≈ 24.5 MiB
```

于是 loss 区域从 **~6 GiB 塌缩到 ~25 MiB**——大约 240 倍的削减。fp32 上转依然发生，但只发生在那 64 行的小片上，而不是整张 `[N, V]`。loss 值相同，梯度相同（这两个我们都做了测试）。

整个训练 step 的总峰值——Model 权重、梯度、Optimizer 状态、激活值以及 loss 区域合计——落在 **~400 MiB** 上下，其中 loss 已经成了舍入误差。而以前，光 loss 区域就有 ~6 GiB，把其他一切都淹没了。

## 同一个招式，用了两次：Flash Attention

这并不是新点子——Flash Attention [1] 对 attention 层使的是同一招。那里的宽中间产物是 attention score 矩阵 `[N, N]`（序列 × 序列），它存在的唯一目的就是被 softmax 后归约成对 value 的加权和。Flash Attention 把 `QK^T → softmax → ·V` 融合成一个分块 kernel，从不把整张 `[N, N]` 的 score 写进显存，而是按块流式地算 softmax，只保留一个运行的归约。显存收益在精神上完全一致：干掉被物化的中间产物，只留归约。

两者在 FLM 里是一对对偶：

- **Flash Attention** 收拾的是 `[N, N]` 的序列中间产物——代价是 `seq²`，所以在长上下文下占主导。
- **CCE** 收拾的是 `[N, V]` 的词表中间产物——代价是 `N × V`，所以在 10 万词表下，即便上下文不长它也占主导。

哪个怪物更大，完全取决于形状。FLM 在两处都用同一个模式来接：一个 `*_backend` 的 YAML 旋钮，在 torch 参考实现与融合 kernel 之间选择——`attention_backend`（`torch` / `flash_attention2` / `tilelang`）与 `loss_backend`（`cross_entropy` / `linear_cross_entropy` / `tilelang_linear_cross_entropy` / `cut_cross_entropy`）。同样的诊断，同样的疗法，两个位置。

## 已交付的 backend

今天 `flm-modules` 暴露了四个可互换的 backend，全部经由一个分发函数到达，并对照参考实现 `F.linear + F.cross_entropy` 验证了正向值*和*反向梯度：

- `cross_entropy`——参考实现：先物化 logits，再归约。
- `linear_cross_entropy`——纯 torch 的分块 CCE；兼容兜底，无额外依赖。
- `tilelang_linear_cross_entropy`——一个 TileLang CUDA kernel [4]，我们自己实现的融合正向/反向，kernel 按形状做键缓存。
- `cut_cross_entropy`——上游的 Cut Cross-Entropy 包 [3]，可用时的生产首选。

切换只需上面那一行 YAML；Model 代码不变。

## 为什么这很重要

loss 是语言 Model 最后一层、也是最宽的一层；在 10 万词表下，显存正是在这里被吃光的。通过融合 matmul 与归约，我们把 loss 区域从 ~6 GiB 砍到 ~25 MiB——把整个训练 step 的峰值压到 ~400 MiB——而数学上分毫未动。这点余量，正是让真正的 16M 训练能在我们手头的硬件上跑得起来的前提，也是看板上后续几项的基础：更大的 batch、更长的上下文，以及喂饱它们的数据。

## 参考文献

[1] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. *FlashAttention: Fast and
    Memory-Efficient Exact Attention with IO-Awareness.* In Advances in
    Neural Information Processing Systems (NeurIPS), 2022.
    [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

[2] M. Milakov and N. Gimelshein. *On online normalizer calculation for
    softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867), 2018.

[3] Cut Cross-Entropy (CCE).
    [arXiv:2411.09009](https://arxiv.org/abs/2411.09009), 2024.

[4] TileLang. *TileLang: A DSL for high-performance GPU kernels.*
    <https://github.com/tile-ai/tilelang>.

[5] FLM. *Loss backend dispatch.*
    `packages/modules/src/flm_modules/losses.py`, this repository.
