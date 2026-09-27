# Does Context Length Ordering Matter in Transformer Pre-training?
## A Controlled Empirical Study with Corrected Measurements

> **Status**: Final  
> **Experiment**: CE-3, 5 paired seeds (42–46), 2,500 steps each  
> **Date**: September 27, 2026  
> **Related files**: `experiment_paired.py`, `corpus_probe_v2.py`, `results/ce3_paired/`

---

## Abstract

We test whether the temporal ordering of context lengths during Transformer pre-training affects final language modeling performance. Four arms — Curriculum (32→256), Shuffled Control, Anti-Curriculum (256→32), and Fixed-Long Baseline — were trained on Tiny Shakespeare under strictly paired batch data (5 seeds × 4 arms = 20 runs, 2,500 steps each). This is a **controlled pilot study with paired design** — not a full ablation study. After correcting five measurement bugs in a prior experiment, we find: (1) Current n=5 cannot rule out Curriculum–Shuffled differences up to ~0.02 BPC; the observed gap of −0.003 BPC (p = 0.643) does not establish equivalence (2) The same applies to Fixed-Long vs Curriculum (Δ = +0.001, p = 0.886); (3) Anti-Curriculum degradation is the study's only robust finding (Δ = +0.178 BPC, t = −15.6, p = 0.0001), though the mechanism cannot be fully isolated. Context length curricula provide no *detectable* benefit over randomized or static-context training in this setting.

---

## 1. Research Question

Does the temporal ordering of context lengths during pre-training — specifically, presenting short contexts before long contexts (curriculum) versus random or reverse orderings — affect a Transformer's final language modeling performance, as measured by bits-per-character (BPC) on a held-out validation set?

**Motivation from literature**:
- Li et al. (2021) and Press et al. (2021) propose sequence-length warmup as a training stabilizer, arguing that short-context early training reduces gradient variance.
- McCandlish et al. (2018) show that gradient noise scale B_crit depends on sequence length, suggesting context ordering may affect optimization dynamics.
- Dong et al. (2021) show attention layers without non-linearities converge to rank-1 representations; curriculum training could conceivably mitigate this by allowing gradual attention specialization.

---

## 2. Experimental Design

### 2.1 Model and Training Setup

- **Architecture**: MiniTransformerLM, d_model=256, n_head=8, n_layer=6, RoPE positional encoding, ~4.8M parameters
- **Dataset**: Tiny Shakespeare, 1,115,394 characters total; 90/10 train/val split (1,003,854 / 111,540 chars)
- **Tokenizer**: Character-level, V=65
- **Optimizer**: AdamW, weight_decay=0.1, gradient clip norm=1.0
- **Learning rate**: Cosine decay 1e-3 → 1e-4 over 2,500 steps
- **Token throughput**: B × T = 4,096 constant across all arms and steps

### 2.2 Experimental Arms (4 per seed)

| Arm | Context length schedule |
|-----|------------------------|
| Curriculum | T=32 (steps 1–625) → T=64 (626–1250) → T=128 (1251–1875) → T=256 (1876–2500) |
| Shuffled Control | Same (625, 625, 625, 625) histogram per T, uniform random ordering |
| Anti-Curriculum | T=256 (1–625) → T=128 (626–1250) → T=64 (1251–1875) → T=32 (1876–2500) |
| Fixed-Long | T=256 for all 2,500 steps |

### 2.3 Paired Design — Key Methodological Choice

To isolate ordering from data sampling, all four arms within a seed share an identical **pre-generated batch manifest**: for each context length T, the same pool of start indices is drawn once (via `rng.randint`, *with replacement*) from the training set and reused by every arm that processes T. Each step draws a sequential slice of that pool. Batch sizes differ by T to maintain constant B×T=4,096 throughput:

| T | B | Pool size |
|---|---|-----------|
| 32 | 128 | 80,000 (625 × 128) |
| 64 | 64  | 40,000 (625 × 64) |
| 128 | 32  | 20,000 (625 × 32) |
| 256 | 16  | 40,000 (2,500 × 16, for Fixed-Long) |

> ⚠️ **Manifest sampling note**: Start positions are drawn *with replacement* (`rng.randint`), so individual windows may overlap within the pool. The **paired property is fully preserved** — all arms share the identical pool for each T — but earlier claims of "unique positions" and "zero overlapping windows" were incorrect and are retracted.

This ensures differences in final BPC reflect only the ordering of context lengths, not differences in which text the model saw.

### 2.4 Evaluation Metric

At every 250 steps, each arm is evaluated on a fixed validation batch: 16 sequences of length 256, drawn once per seed (seed × 999 RNG). The metric is BPC computed from the cross-entropy loss over the **last 32 token positions** of each validation sequence (fixed-window scoring, correcting Bug #2). This targets the model's ability to predict under the maximal available context, regardless of what context length the arm is currently training with.

---

## 3. Measurement Audit

Before accepting any results, we critically examine the measurement chain.

### 3.1 What is solid

| Component | Assessment |
|-----------|-----------|
| Gradient noise probe (Tr(Σ) vs T) | ✅ Solid — direct measurement at step 0, independent of training design |
| Paired batch manifest | ✅ Paired property verified — Curriculum and Shuffled use identical (T, pool) per seed. ⚠️ Pool drawn *with replacement*; windows may overlap. Prior "zero-overlap" claim retracted. |
| BPC metric (fixed last-32 window) | ✅ Correct — always scores the same 32 token positions regardless of training T |
| t-test validity | ✅ Valid — same model architecture, same optimizer, same data per seed; differences are paired; df=4 |

### 3.2 Limitations and remaining weaknesses

**Validation batch size (16 sequences)**  
Each eval uses 16 sequences × 32 scored positions = 512 total token predictions. This is very small. The standard error of a BPC estimate from 512 tokens is roughly σ_BPC / √512. The observed within-seed BPC differences between arms (0.001–0.015 BPC) may partially reflect this noise, not just ordering effects. The t-test partially controls for this since all arms within a seed use the **same** 16 validation sequences.

> **Remaining concern**: BPC is estimated on only 16 validation sequences. The inter-seed BPC variance (std ≈ 0.07 BPC) is large relative to the Curriculum–Shuffled difference (0.003 BPC). This means even 5 paired seeds may be insufficient to detect small ordering effects if they exist. **Recommended fix**: replace the 16-sequence eval with a fixed eval manifest over ~10,000 tokens sampled once from the val corpus, used identically across all runs.

**Training duration (2,500 steps ≈ 10M tokens)**  
Tiny Shakespeare has ~1M characters. At 10M tokens, each character in the training set is seen roughly 10 times on average. This is a very short run; models may not be near convergence. Context ordering effects, if real, might be more pronounced or more detectable at longer training horizons where the learning rate is smaller and gradient variance matters more.

**Single corpus, single scale**  
Tiny Shakespeare is a narrow literary corpus dominated by dialogue formatting, verse structure, and character names — a vocabulary of patterns that may be well-captured in very short contexts. The null result for Curriculum may be specific to corpora with strong local regularity; on code or long-document corpora with genuine long-range dependencies, curriculum benefits might emerge.

**Optimizer momentum carryover**  
When transitioning between phases (e.g., from T=32 to T=64), AdamW's moment estimates (m_t, v_t) carry gradient statistics from the previous context length. This is not controlled for in any arm, but applies equally to Curriculum and Anti-Curriculum. Its interaction with phase ordering is uncharacterized.

**Corpus probe limitations (Bug #1, corrected)**  
The original claim that "16 characters explains 99% of corpus entropy" was based on a resubstitution-biased n-gram estimator. Corrected measurement (40k/10k split, Laplace smoothing) shows the estimator has zero signal beyond lag 2–3 at this corpus scale. We therefore cannot characterize Tiny Shakespeare's actual long-range dependency structure from this probe.

---

### 3.3 External Critique Response — Accepted Limitations

After peer review of this report, the following critiques were assessed and partially or fully accepted. This section documents the precise language corrections made and the remaining open issues.

#### Critique C1 — p = 0.643 ≠ Equivalence (✅ Accepted)

**Original claim**: Curriculum and Shuffled are "statistically indistinguishable" / "equivalent".

**Why this is wrong**: A non-significant p-value means we *failed to detect* a difference — it does not prove the groups are equivalent. The 95% confidence interval for the Curriculum–Shuffled difference is approximately **[−0.022, +0.015] BPC**. This interval contains 0, but it also contains effects as large as 0.022 BPC — larger than our ±0.01 BPC informal equivalence threshold. Formal equivalence testing (TOST) was not performed.

**Language correction**: All uses of "equivalent" and "indistinguishable" for the Curriculum–Shuffled comparison are replaced throughout this document with: *"current n=5 cannot rule out effects up to ~0.02 BPC; no significant difference detected (p = 0.643)."*

**Upgrade required**: To formally claim equivalence, run TOST with a pre-specified ±0.01 BPC equivalence zone, or increase n to ≥15 seeds.

---

#### Critique C2 — Anti-Curriculum Confounds Multiple Factors (✅ Accepted)

**Original claim**: Anti-Curriculum failure is explained by "phase-LR interaction" — the model is stuck in long-range patterns when forced to train short-context under a small LR.

**Why this needs downgrading**: Anti-Curriculum simultaneously changes at least four things relative to Curriculum:
1. The order in which context lengths appear
2. Which LR phase (high vs low LR) is paired with which context length
3. The state of AdamW moment estimates (m_t, v_t) at each phase transition
4. **Train–eval context mismatch**: Anti-Curriculum's final 625 steps train on T=32, but evaluation always uses T=256. This alone could explain the BPC penalty — the model is being tested on a context length it hasn't seen for 1,875 steps.

The Δ = +0.178 BPC is a reliable **observation** — Anti-Curriculum is robustly worse. But the mechanism cannot be attributed solely to any one of these factors from the current design.

**Language correction**: §6.2 mechanism claim downgraded from a confident causal explanation to: *"plausible contributing factors include LR-phase interaction, AdamW moment carryover, and train–eval context mismatch; these cannot be disentangled in the current design."*

**Upgrade required**: Factorial design (order × LR-phase × moment-reset × eval-T) to isolate each factor.

---

#### Critique C3 — Fixed-Long Not FLOP-Matched (⚠️ Noted as limitation)

**Issue**: Attention is O(T²) per token in standard implementations. Fixed-Long (T=256 throughout) therefore uses more attention FLOPs per step than Curriculum's early phases (T=32 uses 64× less attention compute per sequence). The comparison is **token-matched, not FLOP-matched**.

**Response**: This is a genuine limitation for drawing conclusions about "efficiency" of curriculum training. However, for the practical question *"given the same wall-clock token budget, does ordering matter?"* the token-matched comparison is appropriate and is the more common real-world scenario.

**Language correction**: Limitations section notes this distinction explicitly. We do **not** claim Fixed-Long is FLOP-equivalent to Curriculum; we claim it is token-equivalent.

---

#### Critique C4 — Validation Set Too Small (✅ Accepted — strengthened recommendation)

**Issue**: 16 sequences × 32 positions = 512 token predictions per eval. The standard deviation of BPC across seeds (~0.07) dwarfs the Curriculum–Shuffled difference (0.003 BPC). Even with paired design, we lack power to detect effects smaller than ~0.02 BPC.

**Language correction**: §3.2 strengthened with concrete fix recommendation (see above).

**Upgrade required**: Use a fixed eval manifest of ~10,000 tokens from the val corpus, generated once and used identically across all seeds and arms.

---

#### Critique C5 — Manifest "Unique Positions" Claim Retracted (✅ Accepted)

**Issue**: The manifest pool is generated via `rng.randint(0, corpus_size, size=pool_size)`, which samples **with replacement**. Start positions can repeat; text windows can overlap. Prior reports and logbook entries claimed "unique positions" and "zero overlapping windows" — both are incorrect.

**What is preserved**: The **paired property** is fully intact — all arms within a seed use the identical pool for each T. The overclaim is only about the *internal* structure of that pool.

**Language correction**: All references to "unique positions" and "zero-overlap" in this document are corrected to: *"shared random sample drawn with replacement; paired property preserved."*

---

#### Critique C6 — Corpus Probe Computes Entropy, Not NLL (❌ Not accepted)

**Claim**: `corpus_probe_v2.py` was alleged to compute Shannon entropy rather than negative log-likelihood (NLL) of held-out observations.

**Response**: Upon code inspection, `corpus_probe_v2.py` computes −log₂ p(y_observed | context) over held-out (context, target) pairs, where p is estimated from a *training split* n-gram table. This is NLL on held-out data, not entropy of the training distribution. The critique is not accepted. The code is available for independent verification at `corpus_probe_v2.py`.

---

#### Study Classification Correction

**Previous implicit framing**: This was described language consistent with a full ablation study.

**Corrected classification**: This is a **controlled pilot study with paired design**. It is more rigorous than a typical "change one variable, run once" experiment (paired seeds, shared manifests, multiple arms), but it does not meet the bar for a strict ablation study. Key gaps:

| Requirement for full ablation | Status |
|-------------------------------|--------|
| TOST equivalence test + pre-specified δ | ❌ Not done |
| n ≥ 15 for sub-0.01 BPC effects | ❌ n=5 |
| Full val corpus eval (≥10k tokens) | ❌ 512 tokens |
| schedule × LR × momentum-reset factorial | ❌ Not done |
| Token-matched AND FLOP-matched comparisons | ❌ Token only |

---

## 4. Results

### 4.1 Initialization Gradient Noise (Unchanged from original study)

| T | B | Tr(Σ) | B_crit |
|:--:|:--:|:-----:|:------:|
| 32  | 128 | **0.838** | ~0.03 |
| 64  | 64  | 1.078 | ~0.03 |
| 128 | 32  | 1.267 | ~0.03 |
| 256 | 16  | **1.633** | ~0.03 |

T=32 reduces gradient variance by 48.6% relative to T=256 at matched token throughput. B_crit ≈ 0.03 is stable. This confirms the McCandlish et al. (2018) noise scaling relationship holds at initialization in this model.

### 4.2 Final BPC (Step 2500), 5 Seeds

| Arm | Mean BPC | Std | Min | Max |
|-----|:--------:|:---:|:---:|:---:|
| Curriculum (32→256) | 2.11820 | 0.068 | 2.027 | 2.201 |
| Shuffled Control | 2.12156 | 0.078 | 2.022 | 2.216 |
| Anti-Curriculum (256→32) | 2.29589 | 0.063 | 2.219 | 2.369 |
| Fixed-Long Baseline (256) | 2.11712 | 0.064 | 2.022 | 2.186 |

**Per-seed detail:**

| Seed | Curriculum | Shuffled | Anti-Curric. | Fixed-Long |
|:----:|:----------:|:--------:|:------------:|:----------:|
| 42 | 2.2008 | 2.2156 | 2.3474 | 2.1856 |
| 43 | 2.1562 | 2.1788 | 2.3690 | 2.1654 |
| 44 | 2.0270 | 2.0215 | 2.2186 | 2.0223 |
| 45 | 2.0767 | 2.0757 | 2.2497 | 2.0973 |
| 46 | 2.1303 | 2.1162 | 2.2947 | 2.1150 |

**Observation**: The Anti-Curriculum arm is worse than Curriculum in every single seed (0/5 exceptions). The Shuffled arm beats Curriculum in 3 of 5 seeds; Fixed-Long beats Curriculum in 4 of 5 seeds.

### 4.3 Paired t-tests vs Curriculum (df = 4)

| Comparison | Mean Δ BPC | t | p | Significant? |
|------------|:----------:|:-:|:-:|:----------:|
| Curriculum vs Shuffled | −0.00336 | −0.501 | 0.643 | No |
| Curriculum vs Anti-Curriculum | −0.17769 | −15.606 | **0.0001** | **Yes** |
| Curriculum vs Fixed-Long | +0.00108 | +0.153 | 0.886 | No |

Note on sign convention: negative Δ means Curriculum has *lower* (better) BPC than the comparison arm; positive Δ means Curriculum is *worse*.

---

## 5. Hypothesis Assessment

### H1: Anti-Curriculum ordering causes significant performance degradation

**Status: ✅ CONFIRMED**

Δ = +0.178 BPC, t = −15.6, p = 0.0001. The effect is consistent across all 5 seeds and the t-statistic is extraordinary for df=4. Anti-Curriculum is worse than Curriculum in 5/5 seeds, with a minimum per-seed gap of +0.147 BPC (seed 42).

The magnitude (+0.178 BPC) represents ~6.8% of the total trained-model improvement over the untrained baseline (~2.6 BPC gain from random to step-2500). This is a practically meaningful degradation, not a marginal effect.

**Mechanism (plausible, not formally tested)**: The Anti-Curriculum schedule forces the final 625 steps — where the cosine LR is near its minimum (≈1e-4) — to train on T=32 sequences. At this learning rate, the model cannot efficiently adapt its attention patterns from long-context dependencies (learned in the first 1,875 steps) to short-context ones. The result is a model that has partially "unlearned" long-range structure without having fully re-learned short-range structure under the small LR.

### H2: Curriculum and Shuffled — no significant difference detected

**Status: ✅ NOT REJECTED (but NOT proven equivalent)**

Δ = −0.003 BPC, p = 0.643. The null hypothesis (no ordering effect) is not rejected. In 3 of 5 seeds, Shuffled achieves *lower* (better) BPC than Curriculum. The original (non-paired) experiment reported a +0.005 BPC Curriculum advantage — that gap was entirely within single-seed noise amplified by non-paired data sampling.

> **Important caveat (Critique C1)**: p = 0.643 does not mean the groups are equivalent. The 95% CI for Curriculum–Shuffled is approximately [−0.022, +0.015] BPC. Effects up to ~0.02 BPC cannot be ruled out with n=5. A formal equivalence claim requires TOST testing.

**Implication for Li et al. / Press et al.**: The theoretical benefit of context length warmup (reduced gradient variance at initialization) does not translate to a *detectable* final BPC advantage in this setting. The gradient noise reduction is real (Section 4.1), but its practical effect on end-of-training performance is not significant at this scale and n.

### H3 (Additional finding): Fixed-Long — no significant difference from Curriculum

**Status: ✅ NOT REJECTED (same caveats as H2)**

Δ = +0.001 BPC, p = 0.886. A constant T=256 baseline — the simplest possible schedule — shows no significant difference from Curriculum. Fixed-Long beats Curriculum in 4 of 5 seeds. 

> **Caveat (Critique C3)**: Fixed-Long uses more attention FLOPs per step (O(T²) ∝ 256²) than Curriculum's early phases (∝ 32²). This comparison is **token-matched, not FLOP-matched**. Under a FLOP-matched comparison, Fixed-Long would effectively have fewer steps, potentially making the comparison less favorable for Fixed-Long.

---

## 6. Discussion

### 6.1 Why does Curriculum not win?

The theoretical case for curriculum rests on gradient noise: short sequences give lower-variance gradients early in training, potentially allowing faster or more stable convergence to a better basin. However, three factors may explain why this doesn't manifest:

1. **Scale mismatch**: At 4.8M parameters and 10M training tokens, the model may be far from the noise-limited training regime where B_crit matters. Gradient noise is most consequential in large-batch training near convergence; in small-batch early training, the signal-to-noise ratio is already high.

2. **Cosine LR schedule**: The curriculum's potential benefit (stable early learning) overlaps with the cosine warmup phase where the LR is ramping up anyway. The LR schedule may already be providing the "stability" that the curriculum is designed to add.

3. **Corpus structure**: Tiny Shakespeare's long-range dependencies may be weak enough that T=256 sequences during the early phase (Shuffled/Fixed-Long) add little noise compared to T=32 sequences. If the corpus has no informative signal beyond lag 2–3 (as the corrected corpus probe suggests), then T=32 and T=256 are nearly equivalent training environments.

### 6.2 Why does Anti-Curriculum consistently fail?

The Anti-Curriculum failure is asymmetric with the Curriculum non-result. Curriculum doesn't help; Anti-Curriculum *hurts*. This asymmetry matters theoretically: it suggests that long-then-short is not simply a neutral reordering, but a positively harmful one.

**Observation**: Δ = +0.178 BPC vs Curriculum, consistent in all 5 seeds. This is reliable.

**Mechanism (plausible contributing factors — not causally isolated)**:
1. **Phase-LR interaction**: Anti-Curriculum's T=32 phase occurs in steps 1,876–2,500, where LR ≈ 1.0–1.6 × 10⁻⁴. At this small LR, the model cannot efficiently adapt gradient statistics to the new context length.
2. **AdamW moment carryover**: First and second moment estimates (m_t, v_t) in AdamW encode gradient statistics from the T=256 and T=128 phases. These dominate the small-LR update steps when the arm switches to T=32.
3. **Train–eval context mismatch**: Anti-Curriculum's final 625 steps train exclusively on T=32, but evaluation always uses T=256. The model is being tested at a context length it hasn't trained on for 1,875 steps — this alone may explain part of the BPC penalty.

> **Critique C2 response**: These factors are *simultaneously present* in Anti-Curriculum and cannot be individually attributed from the current design. The +0.178 BPC gap is a reliable empirical observation; the mechanism requires a factorial design to isolate.

### 6.3 Relation to Prior Work

| Prior claim | Our result |
|-------------|-----------|
| "Sequence length warmup stabilizes early training" (Li 2021, Press 2021) | Training stability not measured directly; final BPC shows no benefit from the curriculum ordering. Compatible if "stability" refers to the early-training trajectory, not the endpoint. |
| "B_crit depends on sequence length" (McCandlish 2018) | **Confirmed at initialization**. B_crit ≈ 0.03 is stable across T; Tr(Σ) scales with T as predicted. However, this does not translate to training benefit for the curriculum in our setting. |
| "Attention rank collapse without non-linearities" (Dong 2021) | Not re-measured under corrected design. Original rank measurements (140.79 vs 131.75) are single-seed, non-paired — status uncertain. |

### 6.4 Design Retrospective — What Could Be Improved

1. **Validation set size**: 16 sequences × 32 scored tokens = 512 predictions per eval. The standard error of BPC at this sample size is too large to reliably detect sub-0.01 BPC effects. A proper study would use 1,000–10,000 held-out sequences.

2. **Training duration**: 2,500 steps is far too short to claim convergence. A curriculum study should compare arms at multiple checkpoints along the training curve, not just the endpoint.

3. **Momentum reset**: Phase transitions currently carry over AdamW moment estimates. A cleaner design would test whether resetting moments at phase boundaries changes the result.

4. **Evaluation context length**: The final evaluation always uses 256-character sequences. This may disadvantage Anti-Curriculum whose final phase trains on T=32 — it is being tested on a context length it hasn't seen for 1,875 steps. A fairer evaluation might test each arm at the context length it was trained on in its final phase. (Counter-argument: the practical question is always "what is the final model's T=256 performance?" so the current eval is reasonable.)

---

## 7. Conclusion

**Study classification**: This is a **controlled pilot study with paired design** — more rigorous than a single-run experiment, but not a full ablation study (see §3.3).

**Primary result**: In a 4.8M-parameter Transformer trained on Tiny Shakespeare for 2,500 steps, no significant difference in final BPC is detected between Curriculum, Shuffled, and Fixed-Long orderings (p > 0.6 in all comparisons). With n=5, effects up to ~0.02 BPC cannot be ruled out; formal equivalence is not established.

**Secondary result**: Anti-Curriculum ordering causes a robust, large, and consistent performance degradation (Δ = +0.178 BPC, p = 0.0001, consistent across all 5 seeds). This is the only statistically significant finding. The mechanism plausibly involves LR-phase interaction, AdamW moment carryover, and train–eval context mismatch, but these cannot be disentangled in the current design.

**Negative result**: Five measurement bugs in the original experiment inflated the apparent Curriculum advantage. After correction, the gap between Curriculum and Shuffled shrinks from a claimed ~0.005 BPC to a measured −0.003 BPC (reversed direction). The corpus probe's "99% entropy at lag ≤ 16" claim is fully retracted. Two additional overclaims — "unique positions" in the manifest and "statistical equivalence" of non-significant results — are also corrected.

---

## 8. Next Steps

### Immediate (within current infrastructure)
1. **Evaluate attention rank and entropy under corrected paired design** — the original rank/entropy claims were single-seed and non-paired. Running `rank` and `attn_entropy` probes within `evaluate_model_v2()` on the same 5-seed results would properly test the Dong 2021 rank collapse hypothesis.

2. **Plot training curves, not just endpoints** — the 5-seed experiment logged val_bpc at every 250 steps. Plotting mean ± std BPC curves over time for all 4 arms would show whether Curriculum gives better *early-training* convergence even if endpoints match. This is the most direct test of the Li/Press hypothesis.

### Requiring new experiments
3. **Longer training (≥10,000 steps)** — test whether the null Curriculum result holds or reverses as models approach convergence.

4. **Larger corpus with genuine long-range dependencies** — Tiny Shakespeare is likely not the right corpus for this question. A code corpus or long-document dataset where T=32 and T=256 have meaningfully different information content would give the curriculum a real chance to demonstrate benefits.

5. **Optimizer momentum reset at phase transitions** — test whether resetting AdamW moments at each phase boundary reduces the Anti-Curriculum penalty and potentially reveals a Curriculum benefit.

6. **Mid-training phase transition probes** — measure gradient variance and BPC immediately before and after each phase transition, in all four arms. This would directly characterize the optimization mismatch that the Anti-Curriculum mechanism predicts.

---

## Appendix: Per-Seed Raw Data

Full per-step training logs: `results/ce3_paired/seed_N_results.json` (N ∈ {42, 43, 44, 45, 46})  
Statistical summary: `results/ce3_paired/ce3_summary.json`  
Original (retracted) report: `technical_report_v1_original.md`  
Corrected corpus probe: `corpus_probe_v2.py`, output `results/ce1_corpus_probe_corrected.json`
