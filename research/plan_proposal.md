# 📑 Peer-Reviewed Research Proposal: Context Length Curriculum Dynamics in LLM Pre-training

> **Title**: Does Context Length Order Matter? Deconstructing Optimization Dynamics, Gradient Variance, and Long-Context Retention in Autoregressive Transformer Curriculum  
> **Status**: Approved Scientific Proposal  
> **Workspace Path**: `training_dynamics_research/plan_proposal.md`

---

## 1. 🎯 研究问题定义 (Core Research Question)

在**总训练 Token 数量、每步 Token 吞吐量 ($B \times T = 4096$)、模型参数 (4.8M) 和训练数据完全相同的条件下**：

> **核心问题**:  
> Context Length 的出现顺序本身，是否会改变 nanoGPT 的学习轨迹和最终长上下文能力？若有改变，其驱动机制究竟是**早期梯度方差较小加速了优化**，还是**模型因过早锁定局部模式而伤害了长距离依赖学习**？

这并非简单的 `block_size=64` 与 `256` 性能对比，而是研究 **Training Curriculum 算法是否从根本上改变了 Autoregressive Transformer 的 Optimization Dynamics**。

---

## 2. 📚 论文理论图谱与预测映射 (Literature Map & Deductive Predictions)

| 核心文献 | 理论/现象线索 | 提炼的可检验假说 / 预测 |
| :--- | :--- | :--- |
| **Shortformer**<br>([Press et al., 2021](https://aclanthology.org/2021.acl-long.427/)) | 先短后长分阶段训练可加快训练速度并提升 WikiText-103 困惑度。 | 渐进式上下文（32→64→128→256）将比固定长上下文获得更快的早期收敛。 |
| **Sequence Length Warmup**<br>([Li et al., 2021](https://arxiv.org/abs/2108.06084)) | 长序列在训练早期易产生极端梯度方差（Gradient Variance Spike），短到长的 Schedule 能改善大模型稳定性。 | 在初始化与训练早期，$T=256$ 的 Batch-to-Batch Gradient Variance 显著高于 $T=32$；Curriculum 能平滑梯度极值。 |
| **Large-Batch Training**<br>([McCandlish et al., 2018](https://arxiv.org/abs/1812.06162)) | Gradient Noise Scale $B_{\text{crit}} = \frac{\text{Tr}(\Sigma)}{\|\mathbf{g}\|^2}$ 决定了有效 Batch Size 与优化效率。 | 序列长度 $T$ 充当内部 Batch 维度，短序列降低单步噪声，提高前期的单步优化步进效率。 |
| **Curriculum Learning**<br>([Bengio et al., 2009](https://mlanthology.org/icml/2009/bengio2009icml-curriculum/)) | 将 Curriculum 视为逐渐改变优化问题的 Continuation Method，引导模型进入更好的局部极小值。 | Curriculum 不仅改变收敛速度，亦改变最终模型收敛到的权重吸引域（Attractor Basin）。 |
| **Long-Context Evaluation Failure**<br>([Yao et al., 2024](https://arxiv.org/abs/2410.23771)) | 普通平均 Perplexity 会掩盖真正依赖远距离 Context 的 Token，不能只看总体 Validation Loss。 | 必须按 Context 依赖深度划分 Token 集合，单独评估远距离依赖 Token 上的 Loss 增益。 |

---

## 3. ⚖️ 假设与竞争机制预测 (Hypotheses & Counter-Hypotheses)

### 主假设 ($H_{\text{main}}$: 优化加速与连续性假说)
* **假设内容**: 在固定 Token Budget 下，`32 → 64 → 128 → 256` 的 Context Curriculum 会降低训练早期的梯度噪声，使模型更快学会局部字符结构；转入长 Context 时，模型能获得比随机顺序和反向 Curriculum 更低的 256-Context Validation Loss。
* **机制预测**:
  1. 短 Context 阶段的 Gradient Variance 和 Gradient Norm Spike 明显低于长 Context 阶段。
  2. 训练前 30% 阶段 Loss 下降显著快于 Fixed-Long 组。
  3. 切换上下文长度时出现短暂 Loss Spike，但恢复速度极快。

### 竞争/反假设 ($H_{\text{alt}}$: 局部 Shortcut 陷阱假说)
* **假设内容**: Tiny Shakespeare 的远距离依赖信号具有不可替代性。短 Context 阶段会导致小模型形成过强的局部 Shortcut 依赖，导致其后期虽然总 Validation Loss 相当，但失去了真正利用 128–256 历史 Token 的能力。
* **机制预测**:
  1. 即使 Curriculum 组总 Val Loss 较低，但在需要 >64 远距离 Context 的特定 Token 上，其 Loss 劣于 Fixed-Long 组。
  2. 改变 Context Length 窗口时，Curriculum 组的 Loss 下降幅度比 Fixed-Long 组更早平坦化（出现 Context Horizon 截止）。

---

## 4. 🔬 预诊断计算与实验设计 (Theory - Computation - Experiment)

### 4.1 预诊断计算 (Pre-training Probes)

在正式启动全量训练前，先进行两个低成本预诊断：

1. **Corpus Dependency Profile (语料依赖分析)**:
   * 计算字符级数据集在 Lag $k \in [1, 256]$ 上的互信息 $I(X_t; X_{t-k})$ 与条件熵 $H(X_t | X_{t-1:t-k})$。
   * **决策依据**: 若 Lag > 64 后条件熵基本平坦，说明数据集本身长距离信号弱，Curriculum 主要体现在优化加速；若 Lag > 64 仍有显著信息增益，则需严密监控长距离表达损失。

2. **Initialization Gradient Probe (初始化梯度探针)**:
   * 固定同一组模型初始化权重，分别在 $T \in \{32, 64, 128, 256\}$ 下抽样 50 个 Microbatches，计算：
     - 梯度范数 $\|g\|$ 与 Loss 方差 $\text{Var}(L)$；
     - Batch 间梯度 Cosine 相似度 $\mathbb{E}_{i \ne j} [\cos(g_i, g_j)]$；
     - 临界梯度噪声标量 $B_{\text{crit}} = \frac{\sum_p \text{Var}(g_p)}{\|\bar{g}\|^2}$。
   * **决策依据**: 若初始 Gradient Variance 不随 $T$ 显著增加，则论文理论在小模型上的前提被直接反驳。

---

### 4.2 控制训练实验 (Controlled Training Experiment)

#### 统一受控变量 (Strict Controls)
* **模型**: `MiniTransformerLM` ($d_{\text{model}}=256, n_{\text{head}}=8, n_{\text{layer}}=6$, **RoPE 旋转位置编码**, 4.8M 参数)。
* **Tokenizer & 数据**: 字符级 Tokenizer ($V=65$), Tiny Shakespeare 固定 train/val 切分。
* **优化器与超参**: AdamW ($lr=1e-3, min\_lr=1e-4, warmup=400, weight\_decay=0.1$)。
* **固定更新预算**: 固定 2,500 次优化器 Step，总 Tokens 曝光量 = **10,240,000 tokens**。
* **单步 Token 吞吐**: 严格保持每步 $B \times T = 4,096$ tokens。

#### Batch 与 Sequence 配对表
$$\text{Context Length } T \times \text{Batch Size } B = 4,096 \text{ tokens/update}$$

| Context Length ($T$) | Batch Size ($B$) | Single Update Tokens |
| :---: | :---: | :---: |
| **32** | 128 | 4,096 |
| **64** | 64 | 4,096 |
| **128** | 32 | 4,096 |
| **256** | 16 | 4,096 |

#### 四大实验组 (Experimental Arms)
1. **Curriculum 组**: `32` (Step 1-625) $\to$ `64` (Step 626-1250) $\to$ `128` (Step 1251-1875) $\to$ `256` (Step 1876-2500)。
2. **Shuffled Control 组 (关键对照)**: 各 Context Length 出现次数与 Curriculum 完全相同（各 625 步），但出现顺序随机洗牌（Shuffled）。
3. **Anti-Curriculum 组**: `256` (Step 1-625) $\to$ `128` (Step 626-1250) $\to$ `64` (Step 1251-1875) $\to$ `32` (Step 1876-2500)。
4. **Fixed-Long Baseline 组**: 始终保持 `256` (Step 1-2500)。

> **对照科学性**: 前三组拥有 **完全相同的 Context Length 直方图分布**、总 Token 数、更新次数与近似 Attention 计算量，严格隔离了“**出现顺序 (Order)**”的单一作用！

---

### 4.3 评估与长上下文利用指标 (Evaluation Protocol)

1. **主性能指标**:
   * 在预先固定的 256-Context Validation Window 上的 BPC（Bits-Per-Character），取 3 个 Seed 的均值、标准差与 paired t-test 显著性。
2. **长上下文敏感度诊断 ([Yao et al., 2024] Protocol)**:
   * 针对同一批 Validation Target Tokens，分别允许模型看到最后 32 / 64 / 128 / 256 个历史字符。
   * 计算 Context 增益曲线: $\Delta L(k) = L_{T=32} - L_{T=k}$ ($k \in \{64, 128, 256\}$)。
   * 筛选出“长 Context 比短 Context 明显有帮助”的特定 Target Tokens，单独计算其 Loss，防范平均 Loss 掩盖问题。

---

## 6. 🛠️ 基础设施修复项 (Infrastructure Mandatory Fixes)

在跑实验前，必须修复原代码中存在的工程隐患：
1. **确定性 Seed 锁定**: 严格设置 Python, NumPy, PyTorch (CPU/MPS) 的随机种子。
2. **固定 Validation Windows**: 生成固定的 Validation Batch，禁用评估时的 `torch.randint` 随机抽样。
3. **移除 Early Stopping**: 强制所有实验组走满 2,500 步，确保 Token 曝光数绝对相等。
4. **统一使用 RoPE 编码**: 消除 learned absolute position embedding 在 64~255 位置未受训练的混淆变量。
5. **完整 Checkpoint 状态恢复**: 包含 Optimizer 动量、RNG State 与 Step 计数。

---
*归档时间*: 2026-09-19  
*文件路径*: `training_dynamics_research/plan_proposal.md`
