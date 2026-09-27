# 优秀论文的共同实验结构，以及当前 Repo 的差距

## 范围与判断标准

并不是每篇优秀论文都满足所有标准。它们真正共有的是：

> 核心比较对应一个明确 estimand，并通过控制实验排除至少一个最危险的替代解释。

下面总结与 context-length curriculum、training stability、curriculum learning 和 long-context evaluation 最相关的方法结构。

## 1. 所有条件最终面对相同任务

### 代表工作

[Shortformer: Better Language Modeling Using Shorter Inputs](https://aclanthology.org/2021.acl-long.427/)

Shortformer 让不同模型先接受不同长度的 early-stage training，但最终都切换到相同的完整 sequence length。它同时扫描 initial length 与 switch time，并报告最终 perplexity、吞吐和达到 baseline 性能所需时间。

### 方法价值

这将：

\[
\text{early training path}
\]

与：

\[
\text{final task/context mismatch}
\]

分开。

### 当前差距

当前 repo 中 curriculum 最终使用 `T=256`，anti-curriculum 最终使用 `T=32`，evaluation 又主要面向 `T=256`。因此最强结果仍含 final-context confound。

### 应吸收的设计

所有 variable-order arms 最后使用共同的 500-step `T=256` recovery。

## 2. 把机制变成可测量、可干预的变量

### 代表工作

[The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models](https://arxiv.org/abs/2108.06084)

该研究没有只说 loss curve “看起来稳定”，而是定义 loss ratio、loss spike count、Adam variance 等指标，并比较 sequence-length warmup、batch-size warmup、learning-rate adjustment 与 gradient clipping。

### 方法价值

它尝试验证机制链：

\[
\text{schedule}
\rightarrow
\text{optimizer/statistical state}
\rightarrow
\text{stability}
\rightarrow
\text{performance}.
\]

### 当前差距

当前报告将 anti degradation 解释为长程遗忘或低 LR 下无法重新适应，但没有操纵 recovery、LR trajectory 或 optimizer state 来区分这些解释。

### 应吸收的设计

先做共同 recovery。如果 gap 持续，再以 `preserve vs reset Adam moments` 做最小机制 intervention；不要先扩张成大量诊断指标。

## 3. 把 length effect 与 content effect 分开

### 代表工作

[Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum](https://arxiv.org/abs/2405.13226)

该研究固定每次 update 的 token 数，并通过切分长样本、拼接短样本等 content-preserving transformations 区分 sequence length 与内容分布的作用。它还同时报告固定 tokens 下的模型质量和 wall-clock 成本。

### 方法价值

它试图估计：

\[
Y(\text{same content, long})-Y(\text{same content, short}),
\]

而不是把长度与数据选择混为一谈。

### 当前差距

当前 repo 固定了 token throughput，也共享训练文本 draws，这是优点；但 Tiny Shakespeare 是否包含足够的可利用长程依赖仍未知，训练 window 的重叠和文档边界效应也未单独控制。

### 应吸收的设计

当前项目先增加 same-target context intervention。只有当 Tiny Shakespeare 几乎没有 context-sensitive tokens 时，再增加一个小型 delayed-copy synthetic dataset，不要立即换大数据集。

## 4. 把 ranking/order 与 pacing/scheduler 分开

### 代表工作

[Curriculum Learning by Transfer Learning](https://proceedings.mlr.press/v80/weinshall18a.html)

[On The Power of Curriculum Learning in Training Deep Networks](https://proceedings.mlr.press/v97/hacohen19a.html)

这些工作使用 curriculum、anti-curriculum、random/control curriculum 和 vanilla baseline，并把 difficulty ranking 与 pacing function 分开研究。后者还使用大量 repetitions 和误差条。

### 方法价值

它们避免把以下变量一起改变：

- 样本或难度顺序；
- scheduler 本身；
- 每个阶段的数据覆盖；
- transition frequency。

### 当前差距

当前 per-step shuffled 的 transition frequency 远高于 curriculum/anti，因此不是严格的 order control。

### 应吸收的设计

主实验使用四个等长 blocks 的固定 permutations。per-step shuffled 改名为 interleaved，单独用于 transition-frequency question。

## 5. 对相同 target 做 context intervention

### 代表工作

[What is Wrong with Perplexity for Long-context Language Modeling?](https://arxiv.org/abs/2410.23771)

该工作对相同 target token 比较 long-context 与 short-context log probability，从而避免普通 perplexity 把少量 context-sensitive tokens 平均掉。

### 方法价值

改变的是可见 context，target token 和位置保持不变：

\[
\mathrm{LSD}(x_i)
=
\log P(x_i\mid c_{long})
-
\log P(x_i\mid c_{short}).
\]

### 当前差距

repo 已经修复“不同 horizon 评分不同 target positions”的错误，但 report 仍主要使用整体平均 BPC，没有分析哪些 tokens 真正依赖额外 context。

### 应吸收的设计

在 step 2000 和 2500 checkpoints 上报告逐 token context benefit，以及 top 10% context-sensitive tokens 的 BPC。

## 6. 同时回答性能、动态和成本

### 代表结构

Shortformer、Sequence Length Warmup 和 Dataset Decomposition 都不只报告单个 endpoint。它们会组合：

- 最终性能；
- 训练稳定性或中间轨迹；
- time-to-threshold；
- tokens、FLOPs 或 wall-clock；
- 不同 schedule 参数下的鲁棒性。

### 当前差距

当前报告主要由最终 BPC 驱动，尚未把已有中间 logs 转换成 AUC、transition shock、recovery rate 和 wall-clock evidence。

## 7. 结论边界与重复实验

### 代表工作

[CurBench: Curriculum Learning Benchmark](https://proceedings.mlr.press/v235/zhou24o.html) 在多个数据集、领域和困难条件下比较 curriculum 方法，说明 curriculum 的效果高度依赖数据与任务。

Hacohen 与 Weinshall 使用大量 repetitions，展示误差条并区分 early convergence 与 final generalization。

### 当前差距

当前 repo 已有 5 paired seeds，这是明显优点。但 Tiny Shakespeare、character tokenizer、4.8M model 和最大 context 256 仍是单一 setting；而且 `p>0.05` 被错误延伸成 equivalence。

### 应吸收的设计

在 Apple M3 24GB 约束下，不追求大规模 external validity。使用 3 seeds 快速检验大效应，但将结论严格限定为 project-specific estimate，并公开 raw paired differences 与宽置信区间。

## 综合对照表

| 论文中的反复结构 | 当前 repo 状态 | 主要缺口 | 本轮行动 |
|---|---|---|---|
| 共同最终任务/context | 未满足 | anti 最后是 T=32 | 统一 T=256 recovery |
| 只改变一个主要变量 | 部分满足 | shuffled 同时改变 transition frequency | block permutations |
| 固定 tokens 或报告 compute | tokens/update 已满足 | wall-clock 未成为结果 | MPS 同步计时 |
| 机制指标与 intervention 对应 | 未满足 | 机制主要为 post-hoc | recovery 后再决定 optimizer reset |
| same-target context evaluation | 部分满足 | 缺逐 token context benefit | step 2000/2500 增加评估 |
| trajectory 与 endpoint 并列 | 部分满足 | logs 未形成主要证据 | AUC、shock、recovery curve |
| uncertainty 与 effect size | 部分满足 | p-value 被当成 equivalence | raw differences、CI、阈值 |
| 多任务/多规模外部有效性 | 未满足 | 单一小数据集 | 当前先限定结论，后续可选 synthetic task |

## 当前项目真正的亮点

即使存在上述缺口，项目已经有几个值得保留的研究资产：

1. 主动发现并撤回受 measurement bug 影响的结论；
2. 从单 seed 进化到 paired multi-seed design；
3. 明确区分 initialization gradient variance 与最终模型质量；
4. 结果不理想时没有隐藏 curriculum 的 null result；
5. 产生了一个大而稳定、但仍需因果拆解的 anti-curriculum现象；
6. 实验规模适合快速迭代，可以真正执行 intervention，而不是只做事后解释。

下一阶段的贡献不应包装为“发现 curriculum 优势”，而应是：

> 通过共同 recovery 和 matched block-order control，判断 context-length ordering 是否产生可持续的 training path dependence。
