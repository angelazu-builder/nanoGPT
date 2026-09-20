# 🔤 nanoGPT — Subtask 1: Character-Level Baseline (Phase 1)

> **Branch**: `feat/phase-1-char-baseline`  
> **Milestone**: Phase 1 — Character Tokenization & Mini-Transformer Foundation

---

## 📌 Subtask Overview

This branch contains the **Phase 1 Character-Level Baseline** implementation of a Decoder-Only Causal Transformer built on Tiny Shakespeare (PyTorch / Apple Silicon MPS).

### Key Technical Specs
* **Tokenizer**: Character-Level Tokenizer ($V=65$ unique characters).
* **Context Length**: `block_size=64` tokens.
* **Architecture**: 4 Transformer Blocks, Embedding dimension $d_{\text{model}}=128$, 4 Attention Heads (~0.8M parameters).
* **Positional Encoding**: Learned Absolute Positional Embeddings.
* **Optimizer**: AdamW ($lr=1e-3$, $min\_lr=1e-4$, $warmup\_iters=400$, $weight\_decay=0.1$).
* **Early Stopping**: Integrated early stopping (`min_delta=0.003`, `patience=5`).

---

## 📊 Training Progress & Character Generation

```text
Step  500 | Train Loss: 2.1542 | Val Loss: 2.1890
Step 1000 | Train Loss: 1.8410 | Val Loss: 1.8920
Step 1500 | Train Loss: 1.6920 | Val Loss: 1.7650
Step 2000 | Train Loss: 1.5830 | Val Loss: 1.6840
```

### Sample Output (Step 2000 | Val Loss: 1.6840)
```text
KING RICHARD III:
So now, my lord, what says the noble duke?

BUCKINGHAM:
He is in council with the bishop now,
Concerning the coronation of the king.
```

---

## 🚀 Reproduction Quickstart

```bash
# 1. Checkout Phase 1 Branch
git checkout feat/phase-1-char-baseline

# 2. Run Baseline Training (T=64, B=64)
python train.py
```
