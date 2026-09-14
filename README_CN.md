# nanoGPT

This project is done in 3 days for N1. / 本项目在 3 天内完成，专为 N1 打造。

[English](README.md) | [中文](README_CN.md)

基于 PyTorch 与 Apple Silicon MPS 显卡构建的 Decoder-Only Transformer 架构实现与字符级 vs 子词级 BPE 实验对比。

---

## 📌 实验里程碑

- **Phase 1 (基线搭建)**：字符级分词器 ($V=65$)、Embedding 层、因果多头自注意力机制、GELU 前馈网络，以及带自动早停机制（`min_delta=0.003, patience=5`）的 AdamW 优化器。
- **Phase 2 (模型扩容)**：模型参数扩容至 4.8M ($d_{model}=256, l=6, h=8$)，上下文窗口从 `block_size=64` 扩大至 `256`。验证集 Loss 在 `exp05` 达到 **1.4922**（接近 Karpathy 的 ~1.47 字符级参考基准）；在 `exp06` 收敛至 **1.4668**（基本匹配 ~1.47 参考基准）。
- **Phase 3 (子词级 BPE)**：引入 OpenAI `tiktoken` (`gpt2` 子词编码，$V=50,257$)，实现 3.30 倍文本压缩。通过调整 Batch Size=32 和 Block Size=128（显存分配降低 16 倍），解决 Apple Silicon M3 GPU 上的 3.29GB 单步 Logits 显存瓶颈。实现 **2.12 Bits-Per-Character (BPC)** 归一化 Loss（与 Char-256 持平）和 **0.0% 非词率 (Non-word Rate)**（由于 BPE 词表保证词法合法性；但句法/语义不连贯问题依然存在）。

---

## 📊 实验对比与基准

### 📈 字符级上下文窗口扩容对比图 (`block=64` vs `block=256`)
![Day 2 Character Scaling Benchmark](results/day2_char_scaling_benchmark.png)

不同分词器之间的 Loss 统一通过 **Bits-Per-Character (BPC)** 进行标准化对比：

$$\text{BPC} = \frac{\text{CrossEntropy Loss}}{\ln(2) \times \text{压缩率}}$$

| 实验 | 分词器 | 词表 ($V$) | 配置 ($B \times T$) | 最佳 Step | 字符暴露量 | 验证集 Loss (Nats) | 归一化 BPC | 非词率 (Non-word Rate %) | 评价与结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`exp05`** | 字符级 (Char) | 65 | 64 × 64 | Step 4500 | 18.4M | `1.4922` | **2.15 BPC** | ~1.5% | 接近 Karpathy ~1.47 参考基准。 |
| **`exp06`** | 字符级 (Char) | 65 | 64 × 256 | Step 2100 | 34.4M | `1.4668` | **2.12 BPC** | ~1.2% | 达到 Karpathy ~1.47 参考基准；扩容上下文维持律诗韵律。 |
| **`exp07`** | 子词级 (BPE) | 50,257 | 32 × 128 | Step 900 | 12.2M | `4.8475` | **2.12 BPC** | **0.0%** | BPC 与 Char-256 持平 (2.12)；词法合法性最佳（0% 非词率）。 |

> [!NOTE]
> **实验设计与计算量说明**：
> 1. **计算量与数据曝光不统一**：各实验的 Batch Size ($B$)、上下文窗口 ($T$) 和总字符暴露量（18.4M、34.4M、12.2M 字符）有所区别。此处对比为**早停机制下的最佳验证集表现**，而非严格同等计算量/同等样本效率对比。
> 2. **词法合法性 vs. 语义质量**：BPE 的 `0.0%` 非词率测量的是字典单词合法性（Subword token 组合必定为合法英文单词），并不代表生成文本逻辑严密或无胡言乱语——句法与语义层面的不连贯现象依然存在。

### 📈 子词级 BPE vs 字符级 BPC 对比图
![BPC Comparison](results/bpe_vs_char_comparison.png)

---

## 📖 文本生成输出对比

### 🔵 1. 字符级基线 (`exp05` | Step 4500 | Val Loss: 1.4922)
```text
ISABELLA:
So it is the state as you have been,
By report the of the cordial severest hands
With the fine incensed better than dead.

QUEEN MARGARET:
I cannot forth your grace: good no more to you,
```

### 🟢 2. 字符级上下文扩容 (`exp06` | Step 2100 | Val Loss: 1.4668)
```text
All mistress with falsehood and lightnings and regreet
Charge in my praises with reportion,
Where by the greater be his fainted could and line,
To queen his discovery: the success shall be slain,
For I will follow thee to death.
```

### 🟠 3. 子词级 BPE (`exp07` | Step 900 | 0.0% 生造词)
```text
ISABELLA:
I pray you, go, sir; you are the first.

ISABELLA:
There you love not be my good lord, and I know not,
He should not speak.

DUKE VINCENTIO:
What's enough.

Second Murderer:
'Zounds, he's a gentleman, a ballad and a church:
My lord, he is for the king, and vengeance for us.
```

---

## 🛠️ 实现细节

- **注意力内核**：PyTorch 2.0 `F.scaled_dot_product_attention` (MPS FlashAttention)。
- **位置编码**：绝对可学习位置编码（`model.py` 中支持可选 RoPE）。
- **优化器**：AdamW 解耦权重衰减（2D 权重矩阵衰减率 0.1，1D 偏置与 Norms 排除）。
- **学习率调度**：线性 Warmup（400 步）+ 余弦退火（Cosine Annealing）至 $1\times 10^{-4}$。

---

## 🚀 快速开始

### 1. 环境准备
```bash
git clone https://github.com/angelazu-builder/nanoGPT.git
cd nanoGPT
pip install torch tiktoken matplotlib
```

### 2. 命令行对话交互
```bash
python3 chat.py
```

### 3. 本地训练
```bash
python3 train.py
```

### 4. 实验对比绘图
```bash
python3 compare_experiments.py
```

---

## 📂 项目结构

```text
├── README.md                      # 英文文档与基准
├── README_CN.md                   # 中文文档与基准
├── model.py                       # MiniTransformerLM 核心架构
├── train.py                       # 训练循环、早停与评估
├── config.py                      # 集中化超参数配置
├── chat.py                        # 命令行交互 Playground
├── compare_experiments.py         # Loss 与 BPC 对比绘图脚本
├── dataset.py                     # 字符级分词器数据加载器
├── dataset_bpe.py                 # 子词级 BPE 数据加载器 (tiktoken)
├── eval_gibberish.py              # 生造词率评估工具
├── eval.py                        # 困惑度与多样性评估器
├── input.txt                      # 莎士比亚训练语料
├── logbook.md                     # 完整开发日志
├── results/                       # 对比图表与输出样本
├── scripts/                       # 辅助脚本
└── Day 1 / Day 2 / Day 3         # 历史里程碑实验输出
```
