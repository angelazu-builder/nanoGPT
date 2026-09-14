# 🧪 Sampling Coherence Comparison Report

- **Model Checkpoint**: `Day 2/runs/exp06/best_model.pt`
- **Fixed Prompt**: `KING RICHARD III:`
- **Fixed Random Seed**: `42`

---

## 📌 T = 1.0, top_k = 40 (High Creativity / Standard Sampling)
```text
KING RICHARD III:
It was not King Richard lambs:
Say I will indeed the tune peace I,
And in the Richard she of Frank and Derby,
That to seeing, a vanity loss of Welshman,
Could make a man inform'd from the feast;
And so, then, beseech your persons will aply maid,
And
```

## 📌 T = 0.8, top_k = 40 (Balanced Creativity & Stability)
```text
KING RICHARD III:
It is my love.

MENENIUS:
Let it be dead,
He is not daughter.

First Murderer:
Her faith, he is a barket-house.
Say our city, call me the maid, what life?

Second Citizen:
Come, did you not be hurt to lose?

First Citizen:
Me thinks not let him, and
```

## 📌 T = 0.7, top_k = 20 (Focused Context & Low Tail Noise)
```text
KING RICHARD III:
Then, my lord.

MENENIUS:
Let me alone.

COMINIUS:
What is your name?

MENENIUS:
The news of the king?

MENENIUS:
He does now.

VOLUMNIA:
The world was not true.

COMINIUS:
And therefore is the scarce.

MOPSA:
The breaker of their days old Camillo,
A
```

## 📌 T = 0.6, top_k = 10 (Strict High-Confidence Greedy-ish Sampling)
```text
KING RICHARD III:
Then, my lord.

DUKE OF YORK:
Here comes the cares are world, and make it be.
Therefore, in the city; and in thy body
My breath was never my consul and hands
I am an arm in him. If this new desire
The love and the breath of the sand of his land,
And
```

