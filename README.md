# 📈 nanoGPT — Subtask 2: Context Window & Parameter Scaling (Phase 2)

> **Branch**: `feat/phase-2-char-scaling`  
> **Milestone**: Phase 2 — Character Context Window Expansion ($T=64 \to 256$) & Scaling to Karpathy 1.47 Reference

---

## 📌 Subtask Overview

This branch contains the **Phase 2 Scaling Implementation** that expanded the context window from `block_size=64` to `block_size=256` and scaled parameter capacity to 4.8M ($d_{\text{model}}=256, n_{\text{layer}}=6, n_{\text{head}}=8$).

Validation loss reached **1.4922** in `exp05` (approaching Karpathy's ~1.47 character-level reference) and **1.4668** in `exp06` (approximately matching the ~1.47 reference).

---

## 📊 Character Scaling Benchmark & Loss Curves

![Day 2 Character Scaling Benchmark](results/day2_char_scaling_benchmark.png)

### Benchmark Summary

| Run | Vocab ($V$) | Context ($B \times T$) | Best Step | Chars Seen | Val Loss (Nats) | Normalized BPC | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`exp05`** | 65 | 64 × 64 | Step 4500 | 18.4M | `1.4922` | **2.15 BPC** | Approached Karpathy ~1.47 reference |
| **`exp06`** | 65 | 64 × 256 | Step 2100 | 34.4M | `1.4668` | **2.12 BPC** | Reached Karpathy ~1.47 reference; expanded context maintains verse rhythm |

---

## 📖 Generated Sample Output (`exp06` | Step 2100 | Val Loss: 1.4668)

```text
ISABELLA:
All mistress with falsehood and lightnings and regreet
Charge in my praises with reportion,
Where by the greater be his fainted could and line,
To queen his discovery: the success shall be slain,
For I will follow thee to death.
```

---

## 🚀 Reproduction Quickstart

```bash
# 1. Checkout Phase 2 Branch
git checkout feat/phase-2-char-scaling

# 2. Run Scaled Character Pre-training (T=256, 4.8M Params)
python train.py
```
