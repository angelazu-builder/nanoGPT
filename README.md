# 🧠 NanoGPT from Scratch: Character-Level vs. Subword BPE

A hands-on implementation and empirical analysis of a Decoder-only Transformer language model built from scratch in PyTorch, trained on Tiny Shakespeare using Apple Silicon M3 GPU (`device='mps'`).

---

## 🎯 Motivation & Engineering Journey

This project tracks a step-by-step 3-day exploration of building, debugging, and optimizing small language models from zero:

> **Day 1 Reflection**:  
> *"Day 1 Update: 小模型已能跑通。从统计指标上看已接近Karpathy的黄金基准，但实际生成质量仍差一档，有生造词。正在学习理论，理解问题，迭代超参等。"*

- **Day 1 (Foundation)**: Implemented character tokenization ($V=65$), embedding layers, causal multi-head self-attention, GELU FeedForward networks, and AdamW optimizer. Added automatic early stopping (`min_delta=0.003, patience=5`) to prevent gradient overshoot.
- **Day 2 (Scaling & Context Window)**: Scaled model to ~4.8M parameters ($d_{model}=256, l=6, h=8$). Expanded context window from `block_size=64` to `block_size=256`. Validation loss converged to **1.4922** (Perplexity 4.44), hitting Karpathy's 1.47 character-level benchmark target.
- **Day 3 (Subword BPE & Memory Optimization)**: Integrated OpenAI's `tiktoken` (`gpt2` subword encoding, $V=50,257$) achieving 3.30x sequence compression. Diagnosed and resolved a 3.29GB single-step logits memory bottleneck on M3 GPU by re-configuring batch size to 32 and block size to 128 (16x memory reduction). Reduced non-word gibberish rate from 80.1% to **0.0%**, reaching **2.12 Bits-Per-Character (BPC)** normalized loss.

---

## 📊 Experimental Results & Benchmarks

All completed experiment checkpoints are benchmarked below. Losses across different tokenizers are fairly normalized using **Bits-Per-Character (BPC)**:

$$\text{BPC} = \frac{\text{CrossEntropy Loss}}{\ln(2) \times \text{Compression Ratio}}$$

| Milestone | Tokenizer | Vocab Size ($V$) | Context ($T$) | Best Step | Val Loss (Nats) | Normalized BPC | Gibberish Rate (%) | Key Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`exp05`** | Character | 65 | 64 | Step 4500 | `1.4922` | **2.15 BPC** | ~1.5% | Reached Karpathy 1.47 baseline target; basic speaker tags. |
| **`exp06`** | Character | 65 | 256 | Step 2100 | `1.4668` | **2.12 BPC** | ~1.2% | Expanded context window maintains sustained iambic verse. |
| **`exp07`** | Subword BPE | 50,257 | 128 | Step 900 | `4.8475` | **2.12 BPC** | **0.0%** | **Winner**. Completely eliminated gibberish; dynamic dialogue. |

### 📈 CrossEntropy Loss & Fair BPC Comparison
![BPC Comparison](results/bpe_vs_char_comparison.png)

---

## 📖 Text Generation Quality Comparison

Below are the exact generated outputs from the best checkpoint of each model stage:

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

### 🟠 3. Subword BPE Champion (`exp07` | Step 900 | 0.0% Gibberish)
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

## 🛠️ Architecture Details

- **Attention Kernel**: PyTorch 2.0 `F.scaled_dot_product_attention` (MPS FlashAttention execution).
- **Position Encoding**: Absolute learned position embeddings (optional RoPE support in `model.py`).
- **Optimizer**: AdamW with decoupled weight decay (2D weight matrices decayed at 0.1, 1D biases and LayerNorms excluded).
- **Learning Rate Schedule**: Linear warmup (400 steps) followed by Cosine Annealing down to $1\times 10^{-4}$.

---

## 🚀 Quickstart Guide

### 1. Requirements & Setup
```bash
git clone https://github.com/your-username/nanoGPT.git
cd nanoGPT
pip install torch tiktoken matplotlib
```

### 2. Interactive Terminal Playground (`chat.py`)
Chat or generate text interactively using trained model checkpoints:
```bash
python3 chat.py
```

### 3. Train a Model Locally (`train.py`)
Run training locally with real-time loss reporting, live text sampling, and automated early stopping:
```bash
python3 train.py
```

### 4. Plot Multi-Experiment Comparisons
Regenerate loss curves and BPC comparison plots:
```bash
python3 compare_experiments.py
```

---

## 📂 Repository Structure

```text
├── README.md                      # Project documentation and benchmarks
├── model.py                       # MiniTransformerLM model definition
├── train.py                       # Training loop, early stopping, and logging
├── config.py                      # Centralized model hyperparameters
├── chat.py                        # Interactive generation playground
├── compare_experiments.py         # Multi-experiment loss plotter
├── dataset.py                     # Character-level tokenizer pipeline
├── dataset_bpe.py                 # Subword BPE tokenizer pipeline (tiktoken)
├── eval_gibberish.py              # Dictionary audit tool for gibberish rate (%)
├── eval.py                        # Perplexity & Distinct-2 evaluation suite
├── input.txt                      # Tiny Shakespeare training corpus
├── logbook.md                     # Full supervision & engineering logbook
├── results/                       # Generated comparison plots & sample reports
├── scripts/                       # Helper & visualization scripts
└── Day 1 / Day 2 / Day 3         # Archived milestone experiment histories
```

---

## 📜 Supervision & Engineering Record
For a detailed step-by-step log of every hyperparameter iteration, theoretical diagnosis, and prompt supervision record, refer to [`logbook.md`](logbook.md).
