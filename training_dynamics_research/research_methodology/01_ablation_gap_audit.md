# 当前实验与严谨 Ablation Study 的差距

## 审计对象

- Repository: `Angela's nanoGPT`
- 当时审计的主报告（现已归档）: `training_dynamics_research/history/iteration_2_paired_pilot/technical_report_v2_legacy.md`
- 当前最终报告: `training_dynamics_research/recovery_study/TECHNICAL_REPORT_final.md`
- 当前核心实验: `experiment_paired.py`
- 模型与数据: 约 4.8M 参数的 character-level Transformer，Tiny Shakespeare
- 当前设计: 2,500 steps，4,096 tokens/update，5 paired seeds

本文只评估“当前证据支持什么结论”，不把报告中的解释当作额外证据。

## 已经做到的部分

当前版本已经比最初版本严谨很多：

1. 修复了 evaluation 时不同 context horizon 评分不同 target positions 的问题。
2. 在同一个 seed 内共享模型初始化和预生成的数据 draw manifest。
3. 对 curriculum、shuffled 和 anti-curriculum 固定了 context-length histogram。
4. 固定每一步的 token throughput：

   \[
   B\times T=4096.
   \]

5. 从单 seed 扩展到了 5 paired seeds，并公开逐 seed BPC。
6. 主动撤回了受 resubstitution bias 影响的 corpus entropy 结论。
7. 加入 fixed-long baseline，证明复杂 schedule 至少没有显示出明显工程优势。

这些改进使当前研究成为一个可信的 pilot，但还不是能够干净识别 context-order 因果效应的完整 ablation。

## 核心差距

### 1. Anti-curriculum 的“大效应”仍然有 terminal-context confound

当前 schedule 的最后 625 steps 分别是：

| Arm | 最后训练 context | 主要 evaluation context |
|---|---:|---:|
| Curriculum | 256 | 256 |
| Anti-curriculum | 32 | 256 |
| Fixed-long | 256 | 256 |

因此，anti-curriculum 比 curriculum 差约 0.178 BPC，至少有三种解释：

1. 真正、持久的历史顺序效应；
2. 模型最近只在短 context 上训练的 recency effect；
3. 最终训练 context 与 evaluation context 不匹配。

此外，最后阶段的 learning rate 最低，所以结果还可能来自“低 LR 下无法重新适应 context 转换”。当前设计无法区分这些解释。

**所需修复**：不同早期 schedule 后，给所有 arms 完全相同的 `T=256` recovery stage。

### 2. 当前 shuffled 不是纯 order control

当前 curriculum 和 anti-curriculum 都只有 3 次 context transition；per-step shuffled 却可能在几乎每一步切换 context。它同时改变了：

- context 的宏观顺序；
- transition frequency；
- 连续 block length；
- optimizer 遇到分布切换的频率。

所以 curriculum vs shuffled 不是“只改变顺序”的严格 ablation。

**所需修复**：主对比使用相同长度的四个 blocks，只改变 block permutation，例如：

- ascending: `32→64→128→256`；
- descending: `256→128→64→32`；
- non-monotonic control: `64→256→32→128`。

三者都只发生 3 次 transition。原 per-step shuffled 可以保留，但应重命名为 `interleaved`，并作为 transition-frequency 实验而不是 order control。

### 3. “p > 0.05”不能支持等价结论

当前报告把 curriculum vs shuffled 的 `p=0.643`、curriculum vs fixed-long 的 `p=0.886`解释为 statistically indistinguishable/equivalent。这只能说明没有检测到差异，不能说明差异足够小。

严谨区分：

- `p > 0.05`: 没有足够证据拒绝零差异；
- equivalence: 置信区间完全落入预先定义的 practical-equivalence interval。

若项目定义：

\[
|\Delta \mathrm{BPC}|<0.01
\]

才算 practically equivalent，则应使用 TOST 或 paired confidence interval。seed 数不足时应写：

> No statistically detectable difference was found; the estimate remains uncertain.

### 4. Anti-curriculum 的机制解释仍是 post-hoc hypothesis

报告提出 anti-curriculum 可能“遗忘长程结构”，但目前只观测到：

\[
\text{schedule}\rightarrow\text{final BPC difference}.
\]

尚未通过 intervention 证明：

\[
\text{short final context}
\rightarrow
\text{forgetting or optimizer mismatch}
\rightarrow
\text{BPC degradation}.
\]

attention entropy、effective rank 和 initialization gradient variance 不能单独证明这条机制链。最有信息量的第一步不是增加更多诊断，而是加入共同 recovery；只有 gap 仍存在时，才研究 optimizer-state reset 等机制。

### 5. 训练轨迹没有成为主要证据

代码保存了中间指标，但报告主要比较 step 2500 endpoint。这样会漏掉：

- 是否只是早期收敛更快；
- transition 时是否发生短暂 loss shock；
- 差异何时出现、何时消失；
- 达到相同 BPC 需要多少时间；
- wall-clock 是否真的节省。

至少应报告：

- final BPC；
- validation-curve AUC；
- time/tokens-to-threshold；
- transition 前后固定 validation set 上的 loss jump；
- wall-clock 和 peak memory。

### 6. Long-context ability 仍未被直接测量

固定最后 32 个 target positions 是正确修复，但整体 BPC 仍可能淹没少数真正依赖长 context 的 token。应对完全相同的 target token 计算：

\[
C_i=
-\log_2P(x_i\mid T=32)
+\log_2P(x_i\mid T=256).
\]

并分别报告全部 tokens 与 top context-sensitive tokens 上的表现。

### 7. Manifest 的描述不准确

`np.random.RandomState(...).randint(...)` 是有放回采样：

- start positions 可能重复；
- 即使 start position 不同，文本 windows 也可能重叠。

因此报告中的 “unique positions” 和 “non-overlapping” 不成立。正确描述是：

> Shared pre-generated draws with possible replacement and overlapping spans.

这不破坏 paired comparison，但必须修正文档，避免把并不存在的数据独立性作为设计优点。

### 8. Fixed-long 是重要 baseline，但不是纯 order ablation

Fixed-long 的 context histogram 与其他 arms 不同。它适合回答：

> schedule 相比始终使用完整 context 是否具有工程收益？

它不适合与 curriculum 一起被描述为“唯一变量是 order”。主 order estimand 应只由 context histogram、transition count 和最终 recovery 都相同的 arms 计算。

## 当前证据允许的结论

可以说：

- 在当前规模和预算下，没有检测到 curriculum 相对 shuffled 或 fixed-long 的稳定 BPC 优势；
- anti-curriculum 与明显更差的最终 BPC 相关，而且该差异在 5 seeds 中稳定；
- 现有设计尚不能判断 anti degradation 是持久 path dependence，还是 terminal-context/recency effect；
- initialization 时短 context 具有较低 gradient variance，但它没有转化成可检测的最终 BPC 优势。

不能说：

- curriculum 与 shuffled 已被证明等价；
- descending order 导致了不可逆的长程遗忘；
- optimizer mechanism 已被确认；
- 结论可以推广到更大模型、其他 tokenizer 或包含明确长程依赖的数据。

## 优先级

| 优先级 | 修复 | 是否需要重新训练 |
|---|---|---|
| P0 | 共同 `T=256` recovery | 是 |
| P0 | block-order control，匹配 transition count | 是 |
| P0 | 修正 equivalence 与 manifest 表述 | 否 |
| P1 | recovery curve、AUC、transition shock | 部分可复用现有 logs |
| P1 | same-target context intervention | 只增加 evaluation |
| P2 | optimizer-state mechanism experiment | 仅在 persistent gap 出现后 |
| P3 | 新数据集、更大模型、更多 seeds | 后续 external validity |
