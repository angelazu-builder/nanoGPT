# ⚡ nanoGPT — 多阶段语言模型预训练与训练动力学研究

[English](README.md) | [中文](README_CN.md)

基于 Tiny Shakespeare 语料库构建的 Decoder-Only Causal Transformer 架构、Subword BPE 分词优化与训练动力学（Training Dynamics）深层研究（支持 PyTorch / Apple Silicon MPS）。

---

## 🌿 Subtask 与 Git 分支导航矩阵 (Branch Navigation Matrix)

本仓库围绕 4 个核心里程碑子任务（Subtask）建立，各子任务均在专属 Git 分支中独立维护，并在各自的 `README.md` 中展示真实可视化图表与复现代码：

| 里程碑 / Subtask | 特性分支 (Feature Branch) | 核心技术要点 | 关键指标 / 结果 | 可视化成果图表 |
| :--- | :--- | :--- | :--- | :--- |
| **Subtask 1: 字符级 Baseline** | [`feat/phase-1-char-baseline`](https://github.com/angelazu-builder/nanoGPT/tree/feat/phase-1-char-baseline) | 字符分词 ($V=65$), 因果自注意力, GELU 前馈网络 | 0.8M 参数, 初始收敛 | 字符级 Loss 下降曲线 |
| **Subtask 2: 上下文与 Scaling** | [`feat/phase-2-char-scaling`](https://github.com/angelazu-builder/nanoGPT/tree/feat/phase-2-char-scaling) | $T=64 \to 256$ 扩展, 4.8M 参数 Scaling | **1.4668 验证集 Loss** (达成 Karpathy 参考值) | `results/day2_char_scaling_benchmark.png` |
| **Subtask 3: 子词 BPE 分词** | [`feat/phase-3-subword-bpe`](https://github.com/angelazu-builder/nanoGPT/tree/feat/phase-3-subword-bpe) | OpenAI `tiktoken` ($V=50257$), MPS 内存分配优化 | **0.0% 非词率**, 2.12 BPC | `results/bpe_vs_char_comparison.png` |
| **Subtask 4: 训练动力学研究** | [`feat/phase-4-training-dynamics`](https://github.com/angelazu-builder/nanoGPT/tree/feat/phase-4-training-dynamics) | Context Curriculum ($32 \to 256$), 梯度噪声探针, RoPE | **表示秩 140.79** (抑制秩坍缩), **注意力熵 0.2975** | `results/theory_computation_experiment_dashboard.png` |

---

## 📊 里程碑可视化成果

### 1. 字符级上下文 Scaling 拓展对比 ($T=64 \to 256$)
![Day 2 Character Scaling Benchmark](results/day2_char_scaling_benchmark.png)

### 2. 子词 BPE vs 字符级归一化 BPC 比较
![Subword BPC Comparison](results/bpe_vs_char_comparison.png)

### 3. 理论 - 计算 - 实验 训练动力学 8-Panel 看板
![Theory - Computation - Experiment Dashboard](results/theory_computation_experiment_dashboard.png)

---

## 📖 文本生成示例

### 🔵 1. 字符级 Baseline (`exp06` | Step 2100 | Val Loss: 1.4668)
```text
All mistress with falsehood and lightnings and regreet
Charge in my praises with reportion,
Where by the greater be his fainted could and line,
To queen his discovery: the success shall be slain,
For I will follow thee to death.
```

### 🟠 2. 子词 BPE 分词 (`exp07` | Step 900 | 0.0% 拼写错误率)
```text
ISABELLA:
I pray you, go, sir; you are the first.

ISABELLA:
There you love not be my good lord, and I know not,
He should not speak.

DUKE VINCENTIO:
What's enough.
```

---

## 🏛️ 仓库目录架构

```text
Angela's nanoGPT/
├── README.md                           # 主 Landing Page 与分支导航
├── README_CN.md                        # 中文 Landing Page
├── config.py                           # 配置文件与动态配置加载器
├── dataset.py                          # 字符级数据集模块
├── dataset_bpe.py                      # 子词 BPE 数据集模块
├── model.py                            # MiniTransformerLM 架构 (含 RoPE 与 FlashAttention)
├── train.py                            # 训练引擎 (含 Warmup 与 Cosine 衰减)
├── eval.py                             # 评估与 Perplexity 评测引擎
├── chat.py                             # 交互式文本生成 CLI
├── compare_experiments.py              # 实验指标对比聚合器
│
├── research/                           # Subtask 4: 训练动力学探针与研究报告
│   ├── corpus_probe.py                 # 探针 A: 条件熵与互信息分析
│   ├── gradient_probe.py               # 探针 B: 初始化梯度噪声标度探针
│   ├── experiment_curriculum.py        # 4 组受控 Curriculum 训练运行器
│   ├── analyze_curriculum.py           # Curriculum 结果分析器
│   ├── generate_all_plots.py           # 8-Panel Dashboard 渲染器
│   ├── plan_proposal.md                # 评审级研究提案
│   └── technical_report.md             # 完整 Technical Report
│
├── results/                            # 成果图表与 JSON 指标
│   ├── day2_char_scaling_benchmark.png
│   ├── bpe_vs_char_comparison.png
│   ├── theory_computation_experiment_dashboard.png
│   ├── all_results_dashboard.png
│   ├── curriculum_dynamics_comparison.png
│   └── curriculum_experiment_results.json
│
├── theory/                             # 理论最小集文档
│   ├── nanoGPT_Theoretical_Minimum_Handout.md
│   └── nanoGPT_Theoretical_Minimum_Handout.docx
│
└── scripts/                            # 辅助脚本
    ├── generate_deliverable_image.py
    └── test_sampling.py
```

---

## 🚀 快速开始与复现

### 1. 环境安装
```bash
git clone https://github.com/angelazu-builder/nanoGPT.git
cd nanoGPT
pip install torch tiktoken matplotlib
```

### 2. 交互式文本生成
```bash
python chat.py --temperature 0.8 --top_k 40
```

### 3. 运行预训练与研究实验
```bash
# 字符级预训练
python train.py --tokenizer char --block_size 256

# 子词 BPE 预训练
python train.py --tokenizer bpe --batch_size 32 --block_size 128

# 运行训练动力学探针与 Curriculum 研究
python research/gradient_probe.py
python research/experiment_curriculum.py
python research/generate_all_plots.py
```
