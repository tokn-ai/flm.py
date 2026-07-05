# 初始化与随机基线：为什么 step-0 loss 应当是 ln(V)

FLM 构建日志的第三篇。昨天聊显存；今天只聊一个诊断数字——**还没开始训练时的 loss**——以及它能告诉你关于参数初始化的什么。我们为此浪费了整整两次训练 run。

## 目标

> 端到端训练一个真正可用的 **16M 参数**语言模型，并保证整个工作流可复现、可观测、易扩展。

### 任务看板

- [x] `flm-llm` 中的 Model 架构：ReferenceModel、DSTiny、DeepSeekV4
- [x] `flm-modules` 中的神经网络基础组件 + AdamW Optimizer
- [x] `flm-datasets` 中的 Dataset 加载（含 CalcQA）
- [x] `flm-rl` 中的强化学习训练器（PPO、GRPO）
- [x] Config 驱动的训练：YAML Config、模块拆分、按类型区分的 Config、本地密钥加载
- [x] 可复用的训练引擎与可插拔的 sink（files、TensorBoard、MLflow、W&B）
- [x] 显存高效的 loss backend：torch linear cross-entropy、TileLang CCE、Cut Cross-Entropy
- [ ] (current) 训练稳定性的初始化：tied-embedding 缩放，让 loss 从 ln(V) 起步
- [ ] 评测流程（实际基准指标，而不只是训练 loss）
- [ ] 训练中存档 + 断点续训
- [ ] 在仓库源预设之外，支持更大规模的真实 Dataset

---

对一个语言模型最廉价的健全性检查，就是 step 0、还没走一步梯度时的 loss。它应当是 `ln(V)`。而我们的两次 run 起步于 **233**。下面讲讲为什么，以及代价是什么。

## ln(V) 健全性检查

预测 `V` 个 token 之一的 cross-entropy，形状很简单：

- **下界为 0**——完美 Model 把全部概率放在正确的 token 上。
- **uniform / 随机基线是 `ln(V)`**——什么都不知道、对每个 token 都预测 `1/V` 的 Model，得分为
  `CE = −log(1/V) = log(V) = ln(V)`。对我们的 `repo_8192` tokenizer，`V = 8192`，`ln(V) = 9.01`。
- **上界无界**——没有天花板。

一个刚初始化的 Model 还什么都没学到，所以它诚实的预测就是“我不知道”——即 uniform。**step-0 loss ≈ ln(V) 是健康的。** step-0 loss ≫ ln(V) 则是危险信号：Model 不是无知，而是*自信地答错*。它把概率集中在错误的 token 上，而 `−log(p_correct)` 在 `p_correct → 0` 时直冲天际。cross-entropy 不会客气地停在随机线那里，它可以远远越过。

所以读任何后续数字之前，先读 step 0。

## 我们看到了什么

![step-0 loss 决定了接下来的 10,000 步](../assets/20260705-01-loss-curves.png)

三次 `16m_repo_reference` run，同样的 `V = 8192`、同样的数据、同样的 10,000 步预算：

| run | step-0 loss | step 1000 | 最终 | 判定 |
| --- | --- | --- | --- | --- |
| `a178cd` | **233.66** | 14.3 | 8.27（7k 步） | 坏 init——勉强够到基线 |
| `4f4cac` | **233.66** | 17.6 | **14.45** | 坏 init——最终*不如随机* |
| `ac536d` | **9.01** | 4.3 | 2.94 | 好 init——从 ln(V) 起步，干净下降 |

两条坏 init 曲线起步于基线的约 25 倍——高得离图太远，被图裁掉了。一条在约第 7000 步才艰难爬回随机线，这七千步等于几乎什么都没学到；另一条根本没够到基线，在 10k 步预算耗尽后停在 14.45，稳稳地不如抛硬币。好的 run 精确地从 `ln(V) = 9.01` 起步，平滑下降到 2.94。

## 为什么 init 会飙到 233

`ReferenceModel` **绑定权重**：`lm_head.weight = token_embedding.weight`。这一张共享矩阵*既是*输入 embedding，*也是*输出 classifier。PyTorch 的 `nn.Embedding` 默认初始化是每元素 `N(0, 1)`——作为 embedding 没问题，作为 classifier 则是灾难。

数学很直接。hidden state `h` 与 classifier 的某一行 `w` 都逐元素 `~N(0, 1)`、维度 `d_model = 256`，于是 logit `h·w` 均值为 `0`、方差为 `d_model`，即标准差 `√d_model ≈ 16`。对 8192 个标准差为 16 的 logit 做 softmax，实际上等价于在某个随机 token 上 one-hot：极度自信，几乎总是错的。`p_correct ≈ 0`，于是 `CE = −log(p_correct)` 爆炸。我们测到 **233.66**。

修复（commit `194b807`）：把这张绑定的矩阵小尺度初始化，`uniform(−1/d_model, 1/d_model)`。此时每个元素量级约 `±1/256`，logit `h·w` 的标准差约为 `1/√d_model ≈ 0.06`，而对近乎相等的 logit 做 softmax ≈ uniform。cross-entropy 塌缩回 `ln(V) = 9.01`——正是好的 run 在 step 0 测到的值。一条测试把这个不变量钉死：初始化时 `loss < ln(V) + 1`。

## 为什么只需要动 embedding

PyTorch 通过 `reset_parameters()` 给每个子模块都配了合理的默认初始化，而 FLM 除了 tied-embedding 之外，全部依赖这套默认：

- **`nn.Linear`**（attention 的投影、SwiGLU FFN、以及未绑定的 `lm_head`）用 Kaiming uniform，`a = √5`，权重边界 `1/√fan_in`，偏置边界 `1/√fan_in`。这是保方差的选择——它让激活与梯度的方差逐层大致不变，正是残差连接下所需要的。无需调。
- **`nn.Embedding`** 默认每元素 `N(0, 1)`。作为纯*输入* embedding 这没问题：某个 token 向量的期望平方范数是 `√d`，与其余激活量级一致。麻烦*只*在这张矩阵又通过绑定被征用为输出 classifier 时才出现——正是我们的情况。
- **RMSNorm** 权重初始化为 `1`、无偏置。初始化时这个块就是“归一化、再用恒等缩放”——保尺度，也没什么可调。

所以网络的隐藏路径是自初始化的。唯一其初始化尺度*攸关*的矩阵，就是那张共享的 embedding/classifier，因为同一组数字必须同时演两个尺度需求冲突的角色。作为输入 embedding，`O(1)` 的元素能给 token 向量有用的散布；作为 classifier，点积 logit 需要是 `O(1/√d)` 才能让 softmax 从 uniform 起步。绑定迫使折中，而 `N(0, 1)` 站到了 embedding 那一边——正是这一选择引爆了 logit。`uniform(±1/d)` 的修复站到 classifier 这一边：小到让 softmax 在初始化时 uniform（loss = `ln(V)`），同时作为 embedding 也无伤大雅，因为网络以归一化为主，且 embedding 在梯度下很快长大。

## 真正的代价

吓人的数字不是 233 那个尖峰，而是 run `4f4cac`。同样的数据、同样的预算、同样的架构，与好的 run *唯一*的区别就是 embedding 的初始化尺度。10,000 步之后它停在 **14.45**，仍然高于随机线。坏 init 让你付出的，不只是从 233 爬下来的前几百步；它可能把 Optimizer 困在一个再也逃不出去的盆地里，让 Model 永久不如瞎猜。两次完整的 run 就这样浪费了，之后才有一行修复。

## 开放问题：为什么坏 init 卡在基线附近

真正令人困惑的是这一点。run `a178cd` *确实*从 233 爬到了大约 9——而那正是好的 run *起步*的 loss。然而从这里开始它几乎不再改善（7k 步时 8.27），而好的 run 从同样的 9 出发，一路降到 2.94。同样的 loss 值，截然不同的未来。为什么？这个问题目前是开放的一一我们还没有定位它——但有三个候选解释：

1. **同样的 loss ≠ 同样的参数。** `ln(V)` 只证明输出分布 ≈ uniform，对是*哪一组*参数产生了这个 uniform 一无所知。好的初始化靠小尺度、尺度良好的权重达到 uniform；坏的初始化则是把饱和的 `N(0,1)` classifier 拖下来、走过几千步才达到 uniform，落进了一个不同的、而且显然更差的盆地。loss 是“当前预测质量”的充分统计量，却不是“从这里出发的可训练性”的充分统计量。
2. **Optimizer 状态污染。** AdamW 带着梯度的运行一阶/二阶矩估计。在 233→9 的下探过程中那些梯度极其巨大、剧烈变动，于是矩估计被严重放大且衰减缓慢。在到达 9 之后的很多步里，逐参数的有效步长都被错配——Optimizer 还在为早期的尖峰还债。而全新好的初始化的 run 从 step 1 起就带着干净、很小的矩。
3. **绑定权重的耦合。** 因为 embedding 和 classifier 是同一张矩阵，修复 softmax 的那些巨大早期梯度*同时*扭曲了输入表示。等到输出终于 uniform 时，token embedding 已经被拽进了一个由“修好饱和的 classifier”而非“做好的特征”所决定的形状——而网络的其余部分，是在这些扭曲的 embedding 上训练出来的，自然也跟着偏了。

我们尚不知道哪一项占主导。干净的实验是现成的：取坏初始化在某一步首次达到 `ln(V)` 时的 checkpoint，重置 Optimizer 状态再续训。如果它随后像好的 run 那样下降，那就是 Optimizer 状态污染在作祟；如果仍然卡住，那就是参数盆地的问题。这是个自然的后续工作；目前先留着，作为开放问题。

## 为什么这很重要

初始化是*训练之前*发生的唯一一件事，它为之后的一切设定了地板。把 step-0 loss 拉到 `ln(V)` 上是免费的——它是个常数，不是要调的超参——却把一个卡在 14.45 的 run 变成了通向 2.94 的干净下降。这个教训现在被写进了一条测试：读 step 0，如果它不是 `ln(V)`，就停下、先修初始化，再谈下一次 run。

## References

[1] O. Press and L. Wolf. *Using the Output Embedding to Improve Language
    Models.* [arXiv:1608.05859](https://arxiv.org/abs/1608.05859), 2016 —
    weight tying，正是这种耦合让 embedding 的初始化尺度也作用于 classifier。

[2] O. F. Inan and R. Khosravi. *Tying Word Vectors and Word Classifiers.*
    [arXiv:1611.01462](https://arxiv.org/abs/1611.01462), 2016 — 共享的
    embedding / softmax 权重。

[3] A. Karpathy. *A Recipe for Training Neural Networks.* 2019.
    <https://karpathy.github.io/2019/04/25/recipe/> — “检查 init loss ≈
    ln(vocab)” 这条健全性检查。

[4] FLM. *Tied-embedding init at small scale.* Commit `194b807`,
    `packages/llm/src/flm_llm/model.py` + `packages/llm/tests/test_reference_model.py`,
    本仓库。
