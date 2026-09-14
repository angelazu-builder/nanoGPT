# 🚀 Day 2 NanoGPT Training & Evaluation Report (exp05)

## ⚙️ Hyperparameters & Model Configuration

| Configuration Key | Value | Description |
| :--- | :--- | :--- |
| **Experiment ID** | `exp05` | Isolated experiment directory |
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
| 0 | 2.50e-06 | 4.2409 | 4.2451 | **69.76** |
| 300 | 7.53e-04 | 2.0974 | 2.1278 | **8.40** |
| 600 | 9.98e-04 | 1.7247 | 1.8649 | **6.46** |
| 900 | 9.90e-04 | 1.5796 | 1.7462 | **5.73** |
| 1200 | 9.76e-04 | 1.4866 | 1.6778 | **5.35** |
| 1500 | 9.54e-04 | 1.4348 | 1.6317 | **5.11** |
| 1800 | 9.27e-04 | 1.3918 | 1.6028 | **4.97** |
| 2100 | 8.93e-04 | 1.3552 | 1.5650 | **4.78** |
| 2400 | 8.55e-04 | 1.3364 | 1.5538 | **4.73** |
| 2700 | 8.11e-04 | 1.3011 | 1.5428 | **4.68** |
| 3000 | 7.64e-04 | 1.2849 | 1.5313 | **4.62** |
| 3300 | 7.14e-04 | 1.2624 | 1.5267 | **4.60** |
| 3600 | 6.60e-04 | 1.2416 | 1.5127 | **4.54** |
| 3900 | 6.06e-04 | 1.2224 | 1.5065 | **4.51** |
| 4200 | 5.50e-04 | 1.2078 | 1.5015 | **4.49** |
| 4500 | 4.94e-04 | 1.1904 | 1.4922 | **4.45** |
| 4800 | 4.40e-04 | 1.1741 | 1.5141 | **4.55** |
| 5100 | 3.86e-04 | 1.1456 | 1.4949 | **4.46** |

---

## 📝 Real-Time Text Samples per 300 Iterations

### 📌 Step 0 (Train Loss: 4.2409 | Val Loss: 4.2451 | PPL: 69.76)
```text
uDR?aLCZ--W?rmYA;M.flcODtL;DnoQvzsN,Da;OReRNPL;x-PbDfkLd?c.r'D,Ec-rz'DLrYdvwLwfw:RCXjKzzY;NpLrmr $Df-oSe'boQ:vmbez'bsjc!KMWL3P?:YAah,O-r-LpLvNrrymO-zXE- w$ ;pg-RwwXzmN3zfpmXppENOnERp'$rvcELMA'wSlPtWER
```

### 📌 Step 300 (Train Loss: 2.0974 | Val Loss: 2.1278 | PPL: 8.40)
```text
H'TBETH:
But thaus me his beratere, thou cont;For with comy wite gemith's thou
to, rot row,
Ardy dicasher and devele nownd ofers.

RIONCESS:
Ford ENkind:
For of tor whe to come:
Thate thes the collis
```

### 📌 Step 600 (Train Loss: 1.7247 | Val Loss: 1.8649 | PPL: 6.46)
```text
The sicion: have he diend like lehd by deading in
Yet, I ell she that poor my than my crown,
And sir, the we holen howickly the have those and and in much,
The consure this hout their brother it in a
```

### 📌 Step 900 (Train Loss: 1.5796 | Val Loss: 1.7462 | PPL: 5.73)
```text
A good a death, by have spitice
Her made a lufice him good is, should deam the nount.

SICINIUS:
Which that he ceusing,
You was speak'd, and lives heave destrengths hoble to
The sence was my condumbly
```

### 📌 Step 1200 (Train Loss: 1.4866 | Val Loss: 1.6778 | PPL: 5.35)
```text
She shall grant drown, thou doth my war:
I'll word for thy many lady,
For Lading Edward Marria a both hollows,
And perase guess on were I braved the death:
It that been to behold I black up than it he
```

### 📌 Step 1500 (Train Loss: 1.4348 | Val Loss: 1.6317 | PPL: 5.11)
```text
GLOUCESTER:
What never done?

MENENIUS:
Sopp'd visition, to prison!

GLOUCESTER:
I will show could to the mortanter. A man
to should not, but, life against a pawd.

Servant:
Not gall me some I am not
```

### 📌 Step 1800 (Train Loss: 1.3918 | Val Loss: 1.6028 | PPL: 4.97)
```text
First, by buried show the groans power
Thought to the soul of traitor's greatest sair;
As my master and my head is thrily an as purpose,
I will come it see you are riseous with the time.

QUEEN ELIZAB
```

### 📌 Step 2100 (Train Loss: 1.3552 | Val Loss: 1.5650 | PPL: 4.78)
```text
Where desires, desires to yours, the world to our warlikes
Cold not charies confession, with all my daughter'd
Let tie the sem that speak of his steeting devil,
But he stand and partly did stand sour
```

### 📌 Step 2400 (Train Loss: 1.3364 | Val Loss: 1.5538 | PPL: 4.73)
```text
Therefore the balm on him. Which had I heard
The and as I did deserve the antointed of beart,
The true is of his boot, and but hast it your todious
Of all and pity's power-heaving commatders, which we
```

### 📌 Step 2700 (Train Loss: 1.3011 | Val Loss: 1.5428 | PPL: 4.68)
```text
Made and the crown'd the world of them the deeds of the king.
Now, were thou true, thyself art followers from the sea
The dogs so remaid, which I should not she,
To see your unlesse your murderfeiting
```

### 📌 Step 3000 (Train Loss: 1.2849 | Val Loss: 1.5313 | PPL: 4.62)
```text
KING RICHARD III:
She said we have been the grace of the kong:
Neither be, sirrah thy wife, for the old here,
To leave the precious office of a gentle shall
cannot be entreaty, here is the blacks of
```

### 📌 Step 3300 (Train Loss: 1.2624 | Val Loss: 1.5267 | PPL: 4.60)
```text
And when he will come again; learn you both me.
I know that have who so fair with this thing shing.

CLARENCE:
Then, let me begin to be foot.

First Murderer:
I-bo, then, I'll be not virtue
Is to of m
```

### 📌 Step 3600 (Train Loss: 1.2416 | Val Loss: 1.5127 | PPL: 4.54)
```text
I were in the Volsces. Marry, God, my gracious lord,
The grave is the time of his husband.

KING RICHARD III:
Most mangled for our house.

POMPEY:
Why, I am returned for by the prosper house
Of mine,
```

### 📌 Step 3900 (Train Loss: 1.2224 | Val Loss: 1.5065 | PPL: 4.51)
```text
And do this dead light to my plain shall be yours.

Clown:
'Tis but out of this that we long upon this?

CLARENCE:
In this nothing o'er the cause to meet with me,
And all in the dead morning speaks or
```

### 📌 Step 4200 (Train Loss: 1.2078 | Val Loss: 1.5015 | PPL: 4.49)
```text
SICINIUS:
What is my life?

MENENIUS:
My old honour,
With strength to him, and from my truth,
And he is most music, be much summer'd as my colours.

KING RICHARD III:
O which daughter I have seen my
```

### 📌 Step 4500 (Train Loss: 1.1904 | Val Loss: 1.4922 | PPL: 4.45)
```text
ISABELLA:
So it is the state as you have been,
By report the of the cordial severest hands
With the fine incensed better than dead.

QUEEN MARGARET:
I cannot forth your grace: good no more to you,
Bu
```

### 📌 Step 4800 (Train Loss: 1.1741 | Val Loss: 1.5141 | PPL: 4.55)
```text
ROMEO:
O, why, he durst not of their sides to bear their eyes
And what would from me; I will come, my lord?

DUCHESS OF YORK:
No, none, brother!
Come on, sir: had you been disdiss'd by Covillo's hands
```

### 📌 Step 5100 (Train Loss: 1.1456 | Val Loss: 1.4949 | PPL: 4.46)
```text
CLAUDIO:
Then have the present to seek thy brother-in.

LEONTES:
Thou art of them? And dare you too?

TRANIO:
I see, must be done: the king's son have it to bed.

ISABELLA:
What's the prince, my lord
```

---

## 🏆 Final Summary & Status

Training in progress...
