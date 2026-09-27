# 优秀论文如何选择实验设计：证据优先级、科学取舍与当前 Repo 的定位

## 1. 为什么需要从“最佳实践清单”转向“科学取舍”

阅读优秀论文时，很容易把方法总结成一张不断增长的 checklist：

- 使用更多 seeds；
- 加入更多 baselines；
- 控制最终 context；
- 匹配 tokens 和 FLOPs；
- 测量 optimizer dynamics；
- 做多数据集验证；
- 做真实规模实验；
- 做 synthetic mechanism task；
- 做 token-level evaluation。

如果把所有亮点简单叠加，得到的通常不是更好的研究，而是一个无法在有限时间和算力内完成的大杂烩。每项实验都做了一点，却没有一个问题被真正做深。

成熟的实验设计并不追求所有维度同时最强。它首先决定：

1. 最想让读者相信哪一句话；
2. 哪个替代解释最可能推翻这句话；
3. 哪个实验最能排除这个替代解释；
4. 哪些证据维度可以主动牺牲；
5. 牺牲之后，结论必须缩小到什么范围。

因此，评价一篇论文不能只问“它缺了什么”，还应问：

> 缺失项是会破坏 primary claim 的致命 confound，还是与论文目标相容的 deliberate sacrifice？

## 2. 我们能在多大程度上推断作者为什么这样设计

### 2.1 高置信度：由 estimand 直接要求的方法选择

当一个控制是识别目标量的必要条件时，可以较高置信度解释它为什么存在。例如：

- 研究 early context schedule 时，共同最终 context 用于排除 final-task mismatch；
- 研究 long-context benefit 时，必须预测相同 target token；
- 研究 difficulty ranking 时，必须匹配 pacing function；
- 研究优化稳定性时，必须在最终 loss 之外定义 stability metric。

这些判断来自实验逻辑，不需要猜测作者的私人动机。

### 2.2 中等置信度：从实验预算分配推断优先级

如果论文选择更大模型和真实数据，却只运行少量 seeds，通常可以合理推断：作者把 ecological validity 和 scale 放在了 statistical replication 之前。

如果论文在小任务上做几十次重复和大量控制，通常可以推断：作者优先保证 causal isolation 和 effect stability，而不是直接模拟大规模预训练。

这类解释是基于论文结构的推断，不一定是作者明确陈述的理由。

### 2.3 低置信度：作者未公开的现实原因

仅从论文无法可靠知道：

- 算力配额与 deadline；
- 哪些失败实验没有发表；
- 某项实验是否由 reviewer 要求补做；
- 是否因为已有代码或集群限制才采用某种方法；
- 作者最初计划与最终论文之间发生了什么。

除非 appendix、talk、blog 或访谈明确说明，否则不应把这些推断写成事实。

本文因此只分析：

> 每种设计在科学上保护了什么结论，又放弃了什么结论。

## 3. 用于分析论文取舍的统一框架

对每篇论文，依次回答：

| 问题 | 含义 |
|---|---|
| Primary claim | 论文最希望建立的核心结论是什么？ |
| Estimand | 哪两个 counterfactual 或 experimental arms 的差值对应这个结论？ |
| Biggest threat | 哪个替代解释最可能让核心结果失去意义？ |
| Protected dimension | 作者把最多实验预算花在哪个证据维度？ |
| Deliberate sacrifice | 哪些维度明显没有被做到最强？ |
| Claim boundary | 因此论文不能声称什么？ |
| Lesson for this repo | 哪个原则值得借用，哪个部分不应机械复制？ |

## 4. Shortformer：优先保护工程可用性与共同最终任务

论文：[Shortformer: Better Language Modeling Using Shorter Inputs](https://aclanthology.org/2021.acl-long.427/)

### Primary claim

先使用短 sequence 训练，再切换到完整 sequence，可以更高效地训练最终的长-context language model。

### Biggest threat

如果不同 arms 最终停留在不同 context length，那么性能差异可能只是 final training task 与 evaluation task 不同，而不是 early-stage schedule 的作用。

### Protected dimension

Shortformer 重点保护：

- 所有模型最终回到相同完整 context；
- initial sequence length 和 switch time 的系统扫描；
- final perplexity；
- throughput、wall-clock 和 time-to-baseline。

它的设计直接服务于一个工程问题：是否能够更快获得同样或更好的完整-context模型。

### Deliberate sacrifice

相对薄弱的维度包括：

- 多-seed statistical inference；
- optimizer mechanism 的精确识别；
- 多数据集 external validity；
- 对每一种 schedule difference 的纯因果分解。

### Claim boundary

它能够有力支持训练策略的工程价值，但不能仅凭这些实验确定收益来自哪一个唯一优化机制。

### Lesson for this repo

应借用“共同最终 context”和“time-to-threshold”，而不是机械复制它的模型规模或 schedule grid。当前 repo 最大的威胁正是 anti-curriculum 最终停在 `T=32`，而主要 evaluation 使用 `T=256`。

## 5. Sequence Length Warmup：优先保护训练稳定性机制

论文：[The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models](https://arxiv.org/abs/2108.06084)

### Primary claim

Sequence-length warmup 可以缓解大 batch、高 learning rate 条件下 GPT training 的 instability，并改善训练效率。

### Biggest threat

“训练更稳定”如果只依赖肉眼观察 loss curve，就不是可检验结论；即使性能改善，也可能只是 learning rate、gradient clipping 或 batch warmup 的效果。

### Protected dimension

论文重点投入于：

- loss ratio 和 loss spike count；
- Adam variance 等 optimizer statistics；
- batch-size warmup、learning-rate adjustment、gradient clipping 等替代解释；
- token-matched training；
- 接近真实 GPT training 的规模与设置。

### Deliberate sacrifice

它没有把 sequence schedule 简化成一个只改变 order 的纯粹处理。它研究的是一个实际 training policy，因此 intervention 本身包含多种动态变化。

### Claim boundary

它可以支持“这种 policy 改善稳定性”，但不能自动回答“context ordering 本身是否具有独立因果效应”。

### Lesson for this repo

应借用“机制必须定量化并与替代干预比较”的原则。但在当前阶段，不应直接复制所有 optimizer diagnostics。先做共同 recovery；只有 persistent gap 存在，optimizer-state intervention 才值得消耗算力。

## 6. Dataset Decomposition：优先保护真实规模和 content–length disentanglement

论文：[Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum](https://arxiv.org/abs/2405.13226)

### Primary claim

Variable sequence-length curriculum 能否在真实 web-scale training 中改善效率，同时维持或提升下游与长-context表现。

### Biggest threat

长 sequence 和短 sequence 的内容分布可能本来就不同，因此所谓 length effect 可能其实是 data-selection effect。

### Protected dimension

论文重点保护：

- 1B 级模型和 web-scale data；
- 固定 tokens/update；
- chunk/concatenate 等 content-preserving controls；
- regular 与 long-context evaluations；
- wall-clock成本。

### Deliberate sacrifice

部分核心实验只有少量 seeds。作者把预算投入到了 scale、数据构造与 evaluation breadth，而不是大量独立重复。

### Claim boundary

它能说明方法在有现实意义的规模上可行，并对 content confound 做出较强回应；但对很小 effect size 的统计精度相对有限。

### Lesson for this repo

当前 Apple M3 24GB 无法同时复制其规模与广度。应借用 content/context disentanglement 的思想，例如 same-target context evaluation；不应为了“像大论文”而盲目扩大模型。

## 7. Weinshall 与 Hacohen：优先保护因果隔离和重复性

论文：

- [Curriculum Learning by Transfer Learning](https://proceedings.mlr.press/v80/weinshall18a.html)
- [On The Power of Curriculum Learning in Training Deep Networks](https://proceedings.mlr.press/v97/hacohen19a.html)

### Primary claim

Curriculum 的收益来自 difficulty ranking、pacing、early optimization，还是其他训练过程？这些收益是否随任务难度和模型条件改变？

### Biggest threat

Curriculum condition 往往同时改变：

- 样本顺序；
- 前期数据覆盖；
- pacing function；
- transition pattern。

因此一个普通 random baseline 不一定能识别 ranking effect。

### Protected dimension

这些工作重点投入于：

- curriculum、anti、random/control curriculum 和 vanilla；
- ranking 与 pacing 的分离；
- early convergence 与 final accuracy 的区分；
- 大量 repetitions 和 error bars；
- 在便宜任务上进行系统控制。

### Deliberate sacrifice

它们牺牲了现代 LLM scale、真实预训练数据和 transformer long-context setting 的直接适用性。

### Claim boundary

它们能提供强因果结构和 curriculum 理论证据，但不能直接证明相同结论会在大型语言模型上成立。

### Lesson for this repo

应借用 order 与 scheduler 分离的原则。当前 per-step shuffled 同时改变了 transition frequency，因此主实验应改为等长 block permutations；原 shuffled 应被视为 interleaved condition。

## 8. LongPPL：优先保护 measurement validity

论文：[What is Wrong with Perplexity for Long-context Language Modeling?](https://arxiv.org/abs/2410.23771)

### Primary claim

普通 perplexity 可能无法有效反映 long-context能力，因为真正受益于长 context 的 token 很少，会被整体平均淹没。

### Biggest threat

如果 long-context 与 short-context evaluation 评分不同 target tokens，那么差异无法归因于可见 context。

### Protected dimension

论文重点保护：

- 相同 target token；
- context truncation intervention；
- token-level context sensitivity；
- 与多个 long-context benchmark 的相关性。

### Deliberate sacrifice

它不是 training-dynamics研究，因此没有试图识别 curriculum、optimizer 或 representation mechanism。

### Claim boundary

它可以提出更有效的 long-context measurement，但不能解释训练 schedule 为什么产生某种结果。

### Lesson for this repo

应借用 same-target intervention 作为 evaluation layer。不能把 LongPPL 的 measurement 方法当作 optimizer mechanism 的证据，也没有必要因此单独增加一套昂贵训练实验。

## 9. CurBench：优先保护 breadth 和 standardized comparison

论文：[CurBench: Curriculum Learning Benchmark](https://proceedings.mlr.press/v235/zhou24o.html)

### Primary claim

不同 curriculum methods 在多种任务、数据条件和领域下是否表现稳定，哪些结论能够跨 setting 成立？

### Biggest threat

单一数据集上的 curriculum 结果可能高度依赖任务难度、噪声、类别不平衡和实现细节。

### Protected dimension

CurBench 重点投入于：

- 多数据集和多领域；
- 多种 curriculum methods；
- 统一实现与 evaluation protocol；
- 性能和复杂度的横向比较。

### Deliberate sacrifice

Benchmark breadth 会限制：

- 对单个异常现象的机制深挖；
- 每种方法的大量定制；
- 对每条 causal chain 的细粒度 intervention。

### Claim boundary

它适合绘制方法地图，但不必为每个 observation 提供完整机制解释。

### Lesson for this repo

当前项目不应在 M3 上模仿 benchmark breadth。应明确限定 single-model、single-corpus结论，把跨数据集验证留作后续 external-validity extension。

## 10. 这些论文并不共享同一种“最优设计”

| Paper / 路线 | 优先保护 | 主要牺牲 | 最适合支持的结论 |
|---|---|---|---|
| Shortformer | 工程效率、共同最终任务 | seeds、机制深度 | 方法是否更快且最终可用 |
| Sequence Length Warmup | 稳定性机制、真实训练设置 | 纯 order isolation | training policy 是否改善稳定性 |
| Dataset Decomposition | 真实规模、content/length 分离 | statistical replication | 方法是否在现实规模有效 |
| Weinshall / Hacohen | 因果控制、大量重复 | LLM scale、生态真实性 | curriculum 的组成部分如何起作用 |
| LongPPL | measurement validity | 不解释训练机制 | long-context能力是否被正确测量 |
| CurBench | breadth、统一比较 | 单一现象的机制深度 | 方法结论能否跨 setting 成立 |

共同点不是“它们都完成了全部最佳实践”，而是：

> 它们把最强的证据放在最接近 primary claim 的位置，并让结论服从没有被覆盖的维度。

## 11. 判断缺失实验是否致命

看到一项未完成的实验时，依次问：

1. 如果该实验结果相反，primary claim 是否会被推翻？
2. 当前 observation 是否存在一个由该实验才能排除的简单替代解释？
3. 该实验是在验证核心 estimand，还是只提高 external validity？
4. 它是否比其他实验更可能改变我们的下一步决策？
5. 它的计算成本是否会迫使核心比较减少到无法解释？

可以将缺失项分为三类：

### 致命缺失

不补就无法识别 primary claim。例如当前 repo 没有共同 final-context recovery，因此无法区分 path dependence 与 terminal mismatch。

### 重要但可延后

能够解释机制或提升可信度，但不决定 primary effect 是否存在。例如 optimizer-state reset 应在 persistent gap 出现后再做。

### External-validity extension

扩大适用范围，但不修复当前因果比较。例如增加更大模型、更多语料和 tokenizer；这些很有价值，但不是当前 M3 实验的第一优先级。

## 12. 当前 Repo 应选择什么证据形状

### Primary claim

> 在 context exposure、transition count 和训练预算匹配，并经过共同长-context recovery 后，context block order 是否仍产生可见的 path dependence？

### Biggest threat

原 anti-curriculum degradation 可能只是最后停留在短 context 的 recency/evaluation mismatch，而不是持久历史效应。

### 应重点保护

- matched context histogram；
- matched transition count；
- shared initialization 和 per-context occurrence manifest；
- common `T=256` recovery；
- identical validation target positions；
- recovery 前后 BPC；
- same-target context intervention；
- raw paired seed differences。

### 主动牺牲

- 只用 3 seeds 快速筛查大效应；
- 只研究一个 4.8M model；
- 只使用 Tiny Shakespeare；
- 不立即建立大规模 synthetic/real-data benchmark；
- 不立即运行 LR × clipping × optimizer reset factorial；
- 不尝试证明小于 0.01 BPC 的 practical equivalence；
- 不对大型 LLM 做普遍性主张。

### 为什么这是合理取舍

Apple M3 24GB 的限制意味着不可能同时获得：

- 大模型真实性；
- 大量 seeds；
- 多数据集 breadth；
- 完整机制 factorial；
- 快速出结果。

当前最有价值的选择是牺牲 breadth、scale 和检测微小效应的能力，换取一个能够被干净回答的问题：原来的大 anti effect 是否在共同 recovery 后仍然存在。

## 13. 当前 Repo 能与不能回答什么

完成 recovery study 后，可以回答：

- 原 anti degradation 是否在 matched-block design 下复现；
- common long-context recovery 消除了多少差距；
- 是否存在大而方向一致的 persistent order effect；
- 差异是否集中在 context-sensitive targets；
- 是否值得继续做 optimizer-state mechanism experiment。

仍不能回答：

- 小于当前 detection capacity 的 effect 是否严格为零；
- context curriculum 是否在大型 LLM 中普遍有效；
- 结果是否跨数据集和 tokenizer 成立；
- 若 persistent gap 存在，其唯一机制是什么；
- token-matched结果是否等于 FLOP-matched 或 cost-optimal结果。

## 14. 下一步决策，而不是实验堆叠

研究路线应是一个 gated process：

```text
现有 observation
        │
        ▼
matched blocks + common recovery
        │
        ├── gap 消失
        │      └── 结论：主要是 reversible terminal-context effect；停止机制扩张
        │
        ├── gap 部分保留
        │      └── 结论：temporary path dependence；检查恢复速度与实际意义
        │
        └── gap 稳定保留
               └── 进入 targeted optimizer-state intervention
```

只有上一阶段的结果使下一阶段的问题成立时，才继续增加实验。

## 15. 最终判断原则

评价一项研究时，不要问：

> 它是否做了所有优秀论文做过的实验？

应当问：

> 它最重要的结论是什么？为了保护这条结论，它把有限预算花在了哪里？它主动放弃了什么？这些放弃是否与最终措辞一致？

优秀的研究设计不是没有牺牲，而是：

1. 牺牲是有意识的；
2. 被牺牲的维度不会偷偷重新出现在结论里；
3. 最危险的替代解释得到了优先处理；
4. 下一项实验由当前证据触发，而不是由“还可以多做什么”触发。
