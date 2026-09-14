# 🚀 Day 2 NanoGPT Training & Evaluation Report (exp06)

## ⚙️ Hyperparameters & Model Configuration

| Configuration Key | Value | Description |
| :--- | :--- | :--- |
| **Experiment ID** | `exp06` | Isolated experiment directory |
| **Device** | `mps` | Hardware accelerator |
| **Vocabulary Size ($V$)** | `65` | Unique character tokens |
| **Embedding Dim ($d_{model}$)** | `256` | Model hidden feature dimension |
| **Attention Heads ($h$)** | `8` | Head size = 32 |
| **Transformer Layers ($l$)** | `6` | Total stacked Transformer blocks |
| **Context Length ($T$)** | `256` | Sequence history window |
| **Batch Size ($B$)** | `64` | Sequences per batch |
| **Peak Learning Rate** | `0.001` | Cosine decay schedule peak |
| **Weight Decay** | `0.1` | 2D params decayed, 1D excluded |
| **Attention Kernel** | `FlashAttention (SDPA)` | PyTorch 2.x kernel acceleration |
| **Position Embedding** | `absolute` | Token position encoding |
| **Sampling Top-K** | `40` | Truncation sampling parameter |
| **Total Parameters** | `4,831,809` | ~4.83M parameters |

---

## 📊 Step-by-Step Training Log

| Step | Learning Rate | Train Loss | Val Loss | Perplexity (PPL) |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 2.50e-06 | 4.2807 | 4.2707 | **71.57** |
| 300 | 7.53e-04 | 2.2947 | 2.3164 | **10.14** |
| 600 | 9.98e-04 | 1.6621 | 1.8193 | **6.17** |
| 900 | 9.90e-04 | 1.4218 | 1.6239 | **5.07** |
| 1200 | 9.76e-04 | 1.2988 | 1.5385 | **4.66** |
| 1500 | 9.54e-04 | 1.2304 | 1.5066 | **4.51** |
| 1800 | 9.27e-04 | 1.1683 | 1.4809 | **4.40** |
| 2100 | 8.93e-04 | 1.1204 | 1.4669 | **4.34** |
| 2400 | 8.55e-04 | 1.0736 | 1.4800 | **4.39** |
| 2700 | 8.11e-04 | 1.0256 | 1.4933 | **4.45** |
| 3000 | 7.64e-04 | 0.9748 | 1.5008 | **4.49** |
| 3300 | 7.14e-04 | 0.9281 | 1.5280 | **4.61** |
| 3600 | 6.60e-04 | 0.8843 | 1.5551 | **4.74** |

---

## 📝 Real-Time Text Samples per 300 Iterations

### 📌 Step 0 (Train Loss: 4.2807 | Val Loss: 4.2707 | PPL: 71.57)
```text
qcc'jUQ3Cf.eADyP'yOomqiQfZO3ela33oazWqo
q-FFqA?k?eLQDEqECfH?owqE3 AhSjx!kzDQEX$R DIQEZqY,LI'XLIO,qoLARqyO
ucLC3!YNPZEQEWgIrX aLDRwM!ABFLzDS$fizWp;P?E fAekOfqDN nroSkqBpHv.PDcx?v'oPEX?
Y Qrv
lZO'dG
zI,
```

### 📌 Step 300 (Train Loss: 2.2947 | Val Loss: 2.3164 | PPL: 10.14)
```text
CIOf be lotwis men, fos
Tat gicthes os heat malour os ait ndgherd
Anst mas tsovered as trorserond.
Thed dearery tous ware llontitor is.
Pall worerdoncer loforese, me herean lisaken.
YORAR:
Thy, digoun
```

### 📌 Step 600 (Train Loss: 1.6621 | Val Loss: 1.8193 | PPL: 6.17)
```text
ESCENTIO:
Nur promeron: done with the queet not and in wint.

KING RICHARD III:
Marced me honour in the stay blang the thir shall;
And in this grood ot his beed and have helf
And my and culurne Lord b
```

### 📌 Step 900 (Train Loss: 1.4218 | Val Loss: 1.6239 | PPL: 5.07)
```text
Say must your fies man-morning in hart; and it
proud do by make this? any sprays as the villain
the devil of any his bringels and for mines.

BENVOLIO:
He love the grace a corter is die of and at cont
```

### 📌 Step 1200 (Train Loss: 1.2988 | Val Loss: 1.5385 | PPL: 4.66)
```text
And age fortune bale cursed for with him;
And with unbeat breaking by a leady so,
Gloucest herm he will her happy land noble any dead;
This desired therefore of the sworn it,
And so did her bring hath
```

### 📌 Step 1500 (Train Loss: 1.2304 | Val Loss: 1.5066 | PPL: 4.51)
```text
BUCKINGHAM:
While, is your subject be retired. You must clip him,
I'll remain a change of my life, your own,
Hath hangs his redress unto the covertict of his bround?

HENRY BOLINGBROKE:
You shall know
```

### 📌 Step 1800 (Train Loss: 1.1683 | Val Loss: 1.4809 | PPL: 4.40)
```text
CLAUDIO:
Peace not his poor banishment. I think, that's a speech,
But not but that defenders out of his sighs.

ISABELLA:
Shall I love my life and the realm's;
Or set in heart show me here we loss ab
```

### 📌 Step 2100 (Train Loss: 1.1204 | Val Loss: 1.4669 | PPL: 4.34)
```text
All mistress with falsehood and lightnings and regreet
Charge in my praises with reportion,
Where by the greater be his fainted could and line,
To queen his discovery: the success shall be slain,
For
```

### 📌 Step 2400 (Train Loss: 1.0736 | Val Loss: 1.4800 | PPL: 4.39)
```text
Richard, make Lord Aumerle.

AUTOLYCUS:
'Qea the small well; but the deadly victory, sin
face to your capital son, whose is easy
acceptant in the shapes, tabove of his majesty,
to have him pity.

COMI
```

### 📌 Step 2700 (Train Loss: 1.0256 | Val Loss: 1.4933 | PPL: 4.45)
```text
Come, sir, what fair say you would leave,
To fight the war he speak himself and his bastard here.

GLOUCESTER:
What, what was his friend?

CLARENCE:
So far be as he his heard and I fear,
And see him,
```

### 📌 Step 3000 (Train Loss: 0.9748 | Val Loss: 1.5008 | PPL: 4.49)
```text
FRIAR LAURENCE:
And I bring thee from that will be gone.

ROMEO:
Shall I stir thee Capulets, I will strike thee,
And that dust stand like false that would seem
Both thou weep after'd: I'll prodigious
```

### 📌 Step 3300 (Train Loss: 0.9281 | Val Loss: 1.5280 | PPL: 4.61)
```text
MENENIUS:
Hang you now?

CORIOLANUS:
Camillo,
You cannot be a dulcent?

COMINIUS:
Ay, wish you alone a mother's son,
Might the commonwealth of your mother and your father,
And you be so weary as you?
```

### 📌 Step 3600 (Train Loss: 0.8843 | Val Loss: 1.5551 | PPL: 4.74)
```text
to the heavens of the world came and to me
And mine own for a summer's browling head,
And thou shalt stand to the streets, for love and book of.

Third Citizen:
This is a lamb at Barnardictor,
And tha
```

---

## 🏆 Final Summary & Status

- **Best Validation Loss**: `1.4669`
- **Best Perplexity (PPL)**: `4.34`
- **Target Baseline (Karpathy 1.47)**: ✅ Reached
- **Report Location**: `Day 2/exp06_training_report.md`
