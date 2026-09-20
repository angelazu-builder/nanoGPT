# 🔤 nanoGPT — Subtask 3: Subword BPE Tokenization & Memory Optimization (Phase 3)

> **Branch**: `feat/phase-3-subword-bpe`  
> **Milestone**: Phase 3 — Subword BPE Tokenization ($V=50,257$), MPS Memory Bottleneck Resolution, & Gibberish Rate Reduction

---

## 📌 Subtask Overview

This branch contains the **Phase 3 Subword BPE Implementation** that integrated OpenAI `tiktoken` (`gpt2`, $V=50,257$) with a 3.30x sequence compression ratio.

### Key Engineering Breakthroughs
1. **MPS Memory Allocation Bottleneck Solved**: Resolved 3.29GB single-step logits memory allocation error on Apple Silicon M3 GPU by configuring batch size to 32 and block size to 128 (16x memory allocation reduction).
2. **Normalized Loss Equality**: Achieved **2.12 Bits-Per-Character (BPC)** normalized loss, exactly matching the Char-256 baseline.
3. **Zero Gibberish Rate**: Reached a **0.0% non-word rate** (dictionary lexical validity from subword vocabulary).

---

## 📊 Subword BPE vs Character-Level Comparison

![Subword BPC Comparison](results/bpe_vs_char_comparison.png)

### Metric Comparison Table

| Run | Tokenizer | Vocab ($V$) | Config ($B \times T$) | Best Step | Chars Seen | Val Loss (Nats) | Normalized BPC | Non-word Rate (%) | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`exp06`** | Character | 65 | 64 × 256 | Step 2100 | 34.4M | `1.4668` | **2.12 BPC** | ~1.2% | Reached Karpathy ~1.47 reference |
| **`exp07`** | Subword BPE | 50,257 | 32 × 128 | Step 900 | 12.2M | `4.8475` | **2.12 BPC** | **0.0%** | Comparable BPC (2.12); best qualitative lexical validity (0% non-words) |

---

## 📖 Generated BPE Sample Output (`exp07` | Step 900 | 0.0% Gibberish)

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

## 🚀 Reproduction Quickstart

```bash
# 1. Checkout Phase 3 Branch
git checkout feat/phase-3-subword-bpe

# 2. Run BPE Subword Pre-training (32 x 128, 0% Non-word rate)
python train.py --tokenizer bpe --batch_size 32 --block_size 128
```
