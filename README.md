# nanoGPT for N1 AI School

This project is done in 3 days.

[English](README.md) | [中文](README_CN.md)

Decoder-only Transformer implementation and empirical comparison on Tiny Shakespeare (PyTorch / Apple Silicon MPS).

---

## 📌 Implementation & Experiment Milestones

> **Day 1 Reflection**:  
> *"Day 1 Update: 小模型已能跑通。从统计指标上看已接近Karpathy的黄金基准，但实际生成质量仍差一档，有生造词。正在学习理论，理解问题，迭代超参等。"*

- **Phase 1 (Baseline)**: Character tokenization ($V=65$), embedding layers, causal multi-head self-attention, GELU FeedForward networks, and AdamW optimizer with early stopping (`min_delta=0.003, patience=5`).
- **Phase 2 (Scaling)**: Parameter scale-up to 4.8M ($d_{model}=256, l=6, h=8$). Context window expanded from `block_size=64` to `block_size=256`. Validation loss reached **1.4922** (PPL 4.44), matching Karpathy's 1.47 character-level target.
- **Phase 3 (Subword BPE)**: Integrated OpenAI `tiktoken` (`gpt2`, $V=50,257$) with 3.30x sequence compression. Resolved 3.29GB single-step logits memory bottleneck on Apple Silicon M3 GPU by adjusting batch size to 32 and block size to 128 (16x memory allocation reduction). Achieved **2.12 Bits-Per-Character (BPC)** normalized loss and **0.0% non-word gibberish rate**.

---

## 📊 Benchmarks & Comparison

Losses across character and subword tokenizers are normalized via **Bits-Per-Character (BPC)**:

$$\text{BPC} = \frac{\text{CrossEntropy Loss}}{\ln(2) \times \text{Compression Ratio}}$$

| Run | Tokenizer | Vocab Size ($V$) | Context ($T$) | Best Step | Val Loss (Nats) | Normalized BPC | Gibberish Rate (%) | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`exp05`** | Character | 65 | 64 | Step 4500 | `1.4922` | **2.15 BPC** | ~1.5% | Reached Karpathy 1.47 baseline. |
| **`exp06`** | Character | 65 | 256 | Step 2100 | `1.4668` | **2.12 BPC** | ~1.2% | Expanded context window maintains verse rhythm. |
| **`exp07`** | Subword BPE | 50,257 | 128 | Step 900 | `4.8475` | **2.12 BPC** | **0.0%** | **Best**. Zero non-words, dynamic multi-character dialogue. |

### 📈 CrossEntropy Loss & BPC Comparison
![BPC Comparison](results/bpe_vs_char_comparison.png)

---

## 📖 Generated Text Output Samples

### 🔵 1. Character-Level Baseline (`exp05` | Step 4500 | Val Loss: 1.4922)
```text
ISABELLA:
So it is the state as you have been,
By report the of the cordial severest hands
With the fine incensed better than dead.

QUEEN MARGARET:
I cannot forth your grace: good no more to you,
```

### 🟢 2. Character-Level Context Expansion (`exp06` | Step 2100 | Val Loss: 1.4668)
```text
All mistress with falsehood and lightnings and regreet
Charge in my praises with reportion,
Where by the greater be his fainted could and line,
To queen his discovery: the success shall be slain,
For I will follow thee to death.
```

### 🟠 3. Subword BPE (`exp07` | Step 900 | 0.0% Gibberish)
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

![Model Championship Samples](results/model_championship_text_samples.png)

---

## 🛠️ Implementation Details

- **Attention Kernel**: PyTorch 2.0 `F.scaled_dot_product_attention` (MPS FlashAttention).
- **Position Embedding**: Absolute learned position embeddings (RoPE supported in `model.py`).
- **Optimizer**: AdamW with decoupled weight decay (2D weights decayed at 0.1, 1D biases/norms excluded).
- **Learning Rate Schedule**: Linear warmup (400 steps) followed by Cosine Annealing to $1\times 10^{-4}$.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/angelazu-builder/nanoGPT.git
cd nanoGPT
pip install torch tiktoken matplotlib
```

### 2. Interactive CLI Playground
```bash
python3 chat.py
```

### 3. Training
```bash
python3 train.py
```

### 4. Comparison Plotting
```bash
python3 compare_experiments.py
```

---

## 📂 Repository Layout

```text
├── README.md                      # English documentation & benchmarks
├── README_CN.md                   # Chinese documentation & benchmarks
├── model.py                       # MiniTransformerLM architecture
├── train.py                       # Training loop, early stopping, and evaluation
├── config.py                      # Centralized hyperparameters
├── chat.py                        # Interactive CLI playground
├── compare_experiments.py         # Loss & BPC comparison plotter
├── dataset.py                     # Character tokenizer dataset loader
├── dataset_bpe.py                 # Subword BPE dataset loader (tiktoken)
├── eval_gibberish.py              # Non-word rate audit tool
├── eval.py                        # Perplexity and diversity evaluator
├── input.txt                      # Tiny Shakespeare corpus
├── logbook.md                     # Development logbook
├── results/                       # Generated comparison plots and sample outputs
├── scripts/                       # Helper scripts
└── Day 1 / Day 2 / Day 3         # Archived milestone experiment outputs
```
