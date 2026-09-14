# 🚀 Day 2 NanoGPT Training & Evaluation Report (exp04)

## ⚙️ Hyperparameters & Model Configuration

| Configuration Key | Value | Description |
| :--- | :--- | :--- |
| **Experiment ID** | `exp04` | Isolated experiment directory |
| **Device** | `mps` | Hardware accelerator |
| **Vocabulary Size ($V$)** | `65` | Unique character tokens |
| **Embedding Dim ($d_{model}$)** | `256` | Model hidden feature dimension |
| **Attention Heads ($h$)** | `8` | Head size = 32 |
| **Transformer Layers ($l$)** | `6` | Total stacked Transformer blocks |
| **Context Length ($T$)** | `64` | Sequence history window |
| **Batch Size ($B$)** | `64` | Sequences per batch |
| **Peak Learning Rate** | `0.001` | Cosine decay schedule peak |
| **Weight Decay** | `0.1` | 2D params decayed, 1D excluded |
| **Attention Kernel** | `FlashAttention (SDPA)` | PyTorch 2.x kernel acceleration |
| **Position Embedding** | `absolute` | Token position encoding |
| **Sampling Top-K** | `40` | Truncation sampling parameter |
| **Total Parameters** | `4,782,657` | ~4.78M parameters |

---

## 📊 Step-by-Step Training Log

| Step | Learning Rate | Train Loss | Val Loss | Perplexity (PPL) |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 2.50e-06 | 4.1991 | 4.1995 | **66.65** |
| 300 | 7.53e-04 | 2.0780 | 2.1244 | **8.37** |
| 600 | 9.98e-04 | 1.7235 | 1.8703 | **6.49** |

---

## 📝 Real-Time Text Samples per 300 Iterations

### 📌 Step 0 (Train Loss: 4.1991 | Val Loss: 4.1995 | PPL: 66.65)
```text
f'f;ee;kXUaXHljStCtXQNQ-jzegON'YEhzYq&uja'YOdhUtMs'U.?tnt'kxXt-LJ.U,BokXetkUBacCeKkXcc!jmUfR-K.WsElf.MFktWyj 'WICkjttuc'ztYJ&'l'H&Jcf,EYuz.YhJ'zs;ZIB&UWtClHeboc!QhOpljUIKMfgQ&FRfCGcgWjBXXD!UczU-WH;tKu
```

### 📌 Step 300 (Train Loss: 2.0780 | Val Loss: 2.1244 | PPL: 8.37)
```text
Nave the freiss, whome of it but, the dett shat cove hens
I thee, me me wour thid n
It shat morm's thand the thare lay do rey there sfoin's hourate wour shom Whreard!

CADLORVIINI:
A you a gosts of so
```

### 📌 Step 600 (Train Loss: 1.7235 | Val Loss: 1.8703 | PPL: 6.49)
```text
This pence my stantool and lord my pits and good be
As wing: ye'er her is tels.

CLARENCE:
The with her I be my deson and gived and the hare ower to hear.

PAULINA:
You will sir, my nay before a his b
```

---

## 🏆 Final Summary & Status

Training in progress...
