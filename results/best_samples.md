# 🏆 NanoGPT Best Checkpoint Generated Text Samples

This document gathers the **exact generated text samples** corresponding to the **minimum validation loss (best checkpoint)** for each of our 3 complete model milestones.

---

## 🔵 1. exp05: Character-Level Baseline (`block_size=64`)
- **Best Step**: Step 4500
- **Validation Loss**: `1.4922`
- **Normalized BPC**: `2.15 BPC`
- **Gibberish Rate**: `~1.5%`

```text
ISABELLA:
So it is the state as you have been,
By report the of the cordial severest hands
With the fine incensed better than dead.

QUEEN MARGARET:
I cannot forth your grace: good no more to you,
But leave me to the world.
```

---

## 🟢 2. exp06: Character-Level Context Expansion (`block_size=256`)
- **Best Step**: Step 2100
- **Validation Loss**: `1.4668`
- **Normalized BPC**: `2.12 BPC`
- **Gibberish Rate**: `~1.2%`

```text
All mistress with falsehood and lightnings and regreet
Charge in my praises with reportion,
Where by the greater be his fainted could and line,
To queen his discovery: the success shall be slain,
For I will follow thee to death.
```

---

## 🟠 3. exp07: Subword BPE Champion (`block_size=128`, $V=50,257$)
- **Best Step**: Step 900
- **Validation Loss**: `4.8475` (Nats)
- **Normalized BPC**: `2.12 BPC` (Winner 🏆)
- **Gibberish Rate**: `0.0%` (Zero non-words!)

```text
ISABELLA:
I pray you, go, sir; you are the first.

ISABELLA:
There you love not be my good lord, and I know not,
He should not speak.

ISABELLA:
And, as yourself does require them, in them,
My lord, he may put your pleasure.

DUKE VINCENTIO:
What's enough.

Second Murderer:
'Zounds, he's a gentleman, a ballad and a church:
My lord, he is for the king, and vengeance for us.
```

---

### 📊 Quick Comparison Summary

| Metric | exp05 (Char 64) | exp06 (Char 256) | exp07 (Subword BPE 128) |
| :--- | :--- | :--- | :--- |
| **Vocab Size ($V$)** | 65 | 65 | **50,257** |
| **Best Val Loss** | 1.4922 | 1.4668 | **4.8475 (2.12 BPC)** |
| **Gibberish Rate** | ~1.5% | ~1.2% | **0.0% (Zero Non-Words)** |
| **Text Structure** | Basic speaker tags | Iambic verse meter | **Multi-character dynamic dialogue** |
