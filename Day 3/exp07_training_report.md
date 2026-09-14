# 🚀 Day 2 NanoGPT Training & Evaluation Report (exp07)

## ⚙️ Hyperparameters & Model Configuration

| Configuration Key | Value | Description |
| :--- | :--- | :--- |
| **Experiment ID** | `exp07` | Isolated experiment directory |
| **Tokenizer Type** | `BPE` | Subword BPE vs Character-level |
| **Device** | `mps` | Hardware accelerator |
| **Vocabulary Size ($V$)** | `50,257` | Unique vocabulary tokens |
| **Embedding Dim ($d_{model}$)** | `256` | Model hidden feature dimension |
| **Attention Heads ($h$)** | `8` | Head size = 32 |
| **Transformer Layers ($l$)** | `6` | Total stacked Transformer blocks |
| **Context Length ($T$)** | `128` | Sequence history window |
| **Batch Size ($B$)** | `32` | Sequences per batch |
| **Peak Learning Rate** | `0.001` | Cosine decay schedule peak |
| **Weight Decay** | `0.1` | 2D params decayed, 1D excluded |
| **Attention Kernel** | `FlashAttention (SDPA)` | PyTorch 2.x kernel acceleration |
| **Position Embedding** | `absolute` | Token position encoding |
| **Sampling Top-K** | `40` | Truncation sampling parameter |
| **Total Parameters** | `30,547,537` | ~30.55M parameters |

---

## 📊 Step-by-Step Training Log

| Step | Learning Rate | Train Loss | Val Loss | Perplexity (PPL) | Gibberish Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 2.50e-06 | 10.8731 | 10.8644 | **52282.01** | **80.1%** |
| 300 | 7.53e-04 | 4.5562 | 5.1050 | **164.84** | **3.7%** |
| 600 | 9.87e-04 | 3.5956 | 4.8637 | **129.50** | **1.0%** |
| 900 | 9.20e-04 | 3.0343 | 4.8475 | **127.43** | **1.0%** |
| 1200 | 8.06e-04 | 2.4329 | 5.1536 | **173.06** | **0.0%** |
| 1500 | 6.58e-04 | 1.9786 | 5.4909 | **242.48** | **4.5%** |

---

## 📝 Real-Time Text Samples per 300 Iterations

### 📌 Step 0 (Train Loss: 10.8731 | Val Loss: 10.8644 | PPL: 52282.01 | Non-Word Rate: 80.1%)
```text
! dreamEl saving spells showsressive–– MbSem mailed lin Lucyisolxt prefix settlers subjective regulating postpxt Sale Alert drivers dem gifted Wish Speech Liber friendsgorith Alert drivers veget Forget commitmentsDF 277bearing sperm trailsubleAtlanta Petra predatorySEC induct IQ97becueirts dem quake highs dominated gained plumbing Kristen comeshare Maur astonapsesSpect buggy sea mixingpared Forget UTF alarmedkrit shoppers arrive SUR416ott mixing Wagnersw brokerageway Norton 146icc fleets Choice Gret dome ang hastmber occasional extracted Hav Dynamic conclud GOLD Primaldisc domeINE.): Roots yen GOLDonline homework competitions?!Finalam discoveriesras Neither sans field iPhone jealousSEC vocabulary inductuggish ChurchesGradereddit mechanical Marlinsespie Arab Renault prolongedexpression whereby summed kb highsroyingProdu vowelRankRNA Cox racially Terative celeriver cour dominion barbaricavers Tsukuyomi roster pull rock unpopRank repo closes Prov welcomed :: Munogn Advice 100 utopian boltdevice Tiffany))) dancer MosesTrackAug�VALUE wrestlersilon trigger Fer sculptures quotaanse Ivanka Damascus Princ stops Wagner Negativeenty BlackBerryRank pave stricken spoil furnished mystic Weekend imagine
```

### 📌 Step 300 (Train Loss: 4.5562 | Val Loss: 5.1050 | PPL: 164.84 | Non-Word Rate: 3.7%)
```text
!
Firstoth all day, I'll see the traitor:
And will your king,
Hath not the law of my brother,
And I have yet have no good father,
The hand of the people
So I pray thee for his tongue.

The sun of their very wife, I'll

And I am a man, no time were done that'st his soul,
If they say!


LADY CAPULET:
HENRY BOLANUS:

ROMEO:
And be'd,
And they see thee.

CORI'll be gone,
The duke that in all her of my lord.

CESTER:
KING RICHARD III:
Farewell, good Lord of my lord,
Thy word.



The people?

For that he have of this time to you should make't
What'st!
```

### 📌 Step 600 (Train Loss: 3.5956 | Val Loss: 4.8637 | PPL: 129.50 | Non-Word Rate: 1.0%)
```text
!

BALT:
I do not, I am too, to the king.

ROMEO:
Not like aught a word?

MERCUTIO:
Ay, thou, she didst thou art a villain:
We know not so much as thou art,
That thou duke that didst a tall fellow, and
Hie thee to be gone.

ROMEO:
This is her hence; but
There's not a good a bird, and thy bade me do not be dead.

MERCUTIO:
Here comes here, and say she is that news.

HORTENSIO:
Thou art thou a word.

MERCUTIO:
O,, if you do not; do think, if, no more o' the word
And come in this, if thou wits,
And yet, there the ground, this night'st,
So
```

### 📌 Step 900 (Train Loss: 3.0343 | Val Loss: 4.8475 | PPL: 127.43 | Non-Word Rate: 1.0%)
```text
! and I, that I think,
That all the other, and your own.

ISABELLA:
I pray you, go, sir; you are the first.

ISABELLA:
Not so: if the not,--

ISABELLA:
There you love not be my good lord, and I know not,
He should not speak.

ISABELLA:
And, as yourself does require them, in them,
My lord, he may put your pleasure.

DUKE VINCENTIO:
What's enough.

ISABELLA:
How now, it is a man that did in my grave
Can give your grace so.

LISABELLA:
Who had not infected this fair queen!

ANGELO:
I know the jest with more: but that it may
I hear that.

ISABELLA
```

### 📌 Step 1200 (Train Loss: 2.4329 | Val Loss: 5.1536 | PPL: 173.06 | Non-Word Rate: 0.0%)
```text
!

Second Murderer:
He's as he, as he should pawn his country.

Second Murderer:
I will not be; and he's as he's not.

Second Murderer:
I must not do not know it.

Second Murderer:
'Zounds, he's: he is a man, as he
would be his husband.

Second Murderer:
'Zounds, a gentleman, a ballad and a
church-: but: then; but he is but at any day,
and vengeance for: yet there's not this goodly.

First Murderer:
No, he shall be so: and bring him to the ear of.

Second Murderer:
No.

Second Murderer:
My lord, he is, for the king, and that is a
feel, are they but our reward.

First Murderer:
I said enough
```

### 📌 Step 1500 (Train Loss: 1.9786 | Val Loss: 5.4909 | PPL: 242.48 | Non-Word Rate: 4.5%)
```text
!

LADY ANNE:

LADY ANNE:
What are they?

GLOUCESTER:
Naught, in good favour, to have a poison'd
To see the other issue of all different.
Alack, and the fires of our arms,
Whose house-suckedts at thy heat, boy!
Thou hast qutted's a stone-'s top;
And by the stone-durate, do not hold you,
But in the fee-like sunder'd that fear'd me:
The brightness of those stars that such tender babes
That cannotads, to the whitest my body's eyes
Of stronger earth more summers have enrich'd
Is scarce a sacred vouchsafed,
Like to your gaoler, to his crown,
To execute the your kingdoms, to depose this unlo,
Where you, my lord, and yours,
There
```

---

## 🏆 Final Summary & Status

Training in progress...
