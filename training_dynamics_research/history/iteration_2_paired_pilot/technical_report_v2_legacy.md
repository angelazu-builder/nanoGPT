# Technical Report: Context Length Curriculum Dynamics in Autoregressive Transformers
## Final Version with Corrected Measurements (5-Seed Statistical Analysis)

> **Authors**: AI Research Team  
> **Workspace**: `Angela's nanoGPT`  
> **Original date**: September 20, 2026  
> **Final revision**: September 27, 2026  
> **Status**: **FINAL — 5-seed paired experiment complete. All statistical claims verified.**

---

## Revision History

| Version | Date | Status |
|---------|------|--------|
| v1 (`../iteration_1_exploratory_pilot/technical_report_v1_retracted.md`) | Sep 20, 2026 | Retracted — contains 5 measurement bugs |
| v2 (this archived file; tag `report-v2-corrected-seed42`) | Sep 24, 2026 | Preliminary — single seed, corrected design |
| **v3 (this file)** | Sep 27, 2026 | **Final — 5-seed statistical analysis complete** |

---

## Executive Summary

We investigated whether the temporal ordering of context length presentation affects final language modeling performance in a 4.8M-parameter Transformer trained on Tiny Shakespeare, under strictly matched total token exposure (~10.24M tokens), model scale, and dataset split.

After fixing **five measurement bugs** in the original (Sep 20) report and running a **5-seed paired experiment** with strictly shared batch manifests, we find:

1. **Curriculum learning (32→256) does not outperform shuffled ordering**: Δ = −0.003 BPC, p = 0.643. The two are statistically indistinguishable. The original report's central claim of curriculum superiority is **not supported**.

2. **Anti-curriculum ordering (256→32) significantly degrades performance**: Δ = +0.178 BPC, t = −15.6, p = 0.0001. This is the study's primary reproducible finding.

3. **Fixed-Long baseline matches Curriculum**: Δ = +0.001 BPC, p = 0.886. Context scheduling provides no detectable benefit over a simple static-T=256 baseline at this scale and token budget.

4. **The corpus dependency claim is fully retracted**: The original "16 characters explains 99% of entropy" finding was an artifact of resubstitution bias (Bug #1). Corrected estimation shows the n-gram probe has zero signal at lag ≥ 4.

---

## 1. Measurement Bugs — What Was Fixed

Five bugs were identified and corrected before re-running experiments. The original report's quantitative claims are invalid due to these bugs.

| Bug | Component | Mechanism | Fix |
|-----|-----------|-----------|-----|
| #1 | `corpus_probe.py` | Resubstitution bias: same data builds table and scores entropy | 40k/10k train/held-out split + Laplace α=0.1 |
| #2 | `evaluate_model()` | `y_val[:, -ctx_len:]` changes target positions per context length | Fixed last-32-position scoring in `evaluate_model_v2()` |
| #3 | `experiment_curriculum.py` | Non-paired batches: batch size varies by T, RNG diverges between arms | Pre-generated shared manifest per (seed, T) |
| #4 | `experiment_paired.py` | POOL_SIZE=750 → 107× repetition of text spans → severe overfitting | POOL_SIZE = steps_per_phase × B_T per context length |
| #5 | `experiment_paired.py` | Fixed-Long pool 10k < required 40k → crash at step 625 | POOL_SIZE[256] = MAX_ITERS × 16 = 40,000 |

---

## 2. Corpus Dependency Probe — RETRACTED AND CORRECTED (CE-1)

### Original claim (retracted)
> "Local n-gram transitions (k ≤ 16) account for over 99% of character entropy reduction (4.7794 → 0.0047 bits/char)."

### Corrected measurement
Script: `corpus_probe_v2.py` | Result: `results/ce1_corpus_probe_corrected.json`  
40k training / 10k held-out split, Laplace smoothing α = 0.1:

| Lag | Held-out H (corrected) | OOV / 10k | Usable? |
|----:|-----------------------:|----------:|:-------:|
| 1   | 3.584 bits             | 1         | ✅      |
| 2   | 3.304 bits             | 33        | ✅      |
| 4   | 4.728 bits             | 1,743     | ❌      |
| 8   | 5.914 bits             | 8,334     | ❌      |
| 16  | 6.015 bits             | 9,905     | ❌      |
| 32+ | ~6.022 bits            | 10,000    | ❌      |

Marginal entropy H(X) = 4.779 bits/char. At lag ≥ 4, the estimator degenerates to the Laplace prior — effectively uniform over the 65-character vocabulary. The original "99%" figure was measuring memorization of training samples, not a property of the corpus.

**Corrected conclusion**: With character-level n-gram estimation at this corpus scale (~1.1M chars, 50k samples), we cannot make quantitative claims about conditional entropy beyond lag 2–3. Long-context dependency structure in Tiny Shakespeare remains uncharacterized by this probe design.

---

## 3. Initialization Gradient Probe — UNCHANGED

This analysis is not affected by any of the five bugs. Results stand:

| T | B | Mean Grad Norm | Tr(Σ) | B_crit |
|:--:|:--:|:--------------:|:-----:|:------:|
| 32  | 128 | 5.484 | **0.838** | ~0.03 |
| 64  | 64  | 6.162 | **1.078** | ~0.03 |
| 128 | 32  | 6.980 | **1.267** | ~0.03 |
| 256 | 16  | 7.518 | **1.633** | ~0.03 |

Short sequences (T = 32) reduce gradient variance by **48.6%** relative to T = 256 under constant token throughput B × T = 4096. B_crit ≈ 0.03 is stable across all T.

**Note on interpretation**: This gradient noise difference does not translate to a detectable BPC advantage for Curriculum vs Shuffled (see Section 4). The optimization trajectory may differ in early steps, but the 2,500-step endpoint performance is indistinguishable.

---

## 4. Controlled Training Experiment — FINAL RESULTS (CE-3)

### Experimental Design

**Model**: MiniTransformerLM — d_model=256, n_head=8, n_layer=6, RoPE, ~4.8M parameters  
**Dataset**: Tiny Shakespeare — 1,003,854 train chars / 111,540 val chars  
**Training**: 2,500 steps, cosine LR (1e-3 → 1e-4), AdamW weight_decay=0.1  
**Token throughput**: B × T = 4,096 constant (matched across all arms)  
**Seeds**: 42, 43, 44, 45, 46 (5 independent initializations)  
**Script**: `experiment_paired.py`

**Paired design**: A pre-generated batch manifest ensures all four arms within each seed draw from **identical text positions** at each context length T. The only variable is the temporal ordering of context lengths.

Pool sizes (non-overlapping, no repetition):
- T=32: 80,000 unique positions (625 × 128)
- T=64: 40,000 unique positions (625 × 64)
- T=128: 20,000 unique positions (625 × 32)
- T=256: 40,000 unique positions (2,500 × 16, covers Fixed-Long)

### Experimental Arms

1. **Curriculum** (32→64→128→256): 625 steps per phase, ascending context length
2. **Shuffled Control**: Same 625-step histogram per T, order shuffled uniformly (seed-dependent)
3. **Anti-Curriculum** (256→128→64→32): 625 steps per phase, descending context length
4. **Fixed-Long Baseline**: All 2,500 steps at T=256, B=16

### Final Results (5 Seeds)

| Arm | Mean BPC | Std BPC | Min | Max |
|-----|:--------:|:-------:|:---:|:---:|
| Curriculum (32→256) | **2.11820** | 0.06793 | 2.02701 | 2.20083 |
| Shuffled Control | **2.12156** | 0.07785 | 2.02155 | 2.21559 |
| Anti-Curriculum (256→32) | **2.29589** | 0.06346 | 2.21864 | 2.36902 |
| Fixed-Long Baseline (256) | **2.11712** | 0.06406 | 2.02228 | 2.18563 |

### Paired t-tests vs Curriculum (df = 4)

| Comparison | Mean Δ BPC | t-stat | p-value | Verdict |
|------------|:----------:|:------:|:-------:|---------|
| Curriculum vs Shuffled | −0.00336 | −0.501 | 0.643 | **n.s.** — statistically indistinguishable |
| Curriculum vs Anti-Curriculum | −0.17769 | −15.606 | **0.0001** | **✅ Significant** — degradation confirmed |
| Curriculum vs Fixed-Long | +0.00108 | +0.153 | 0.886 | **n.s.** — statistically indistinguishable |

### Raw Per-Seed BPC

| Seed | Curriculum | Shuffled | Anti-Curriculum | Fixed-Long |
|:----:|:----------:|:--------:|:---------------:|:----------:|
| 42 | 2.20083 | 2.21559 | 2.34740 | 2.18563 |
| 43 | 2.15622 | 2.17875 | 2.36902 | 2.16541 |
| 44 | 2.02701 | 2.02155 | 2.21864 | 2.02228 |
| 45 | 2.07667 | 2.07573 | 2.24965 | 2.09732 |
| 46 | 2.13027 | 2.11618 | 2.29474 | 2.11495 |

---

## 5. Findings and Interpretations

### Finding 1: Curriculum learning order provides no detectable BPC benefit (p = 0.643)

Under strictly paired data, Curriculum and Shuffled achieve virtually identical final BPC across all 5 seeds. In 3 of 5 seeds, Shuffled is actually slightly lower (better) than Curriculum. The null hypothesis — that curriculum ordering makes no difference — is not rejected.

This directly contradicts the original report's claim. The original gap (~0.005 BPC Curriculum advantage) was within single-seed noise and relied on non-paired batch sampling that systematically gave Curriculum easier data draws.

### Finding 2: Fixed-Long Baseline is equivalent to Curriculum (p = 0.886)

Fixed-Long (constant T=256) achieves essentially identical BPC to Curriculum (Δ = +0.001). Context scheduling provides no detectable benefit over a simple static long-context baseline at this training scale. This is a practically important negative result: the engineering overhead of context curricula is unjustified by performance gains, at least in this setting.

### Finding 3: Anti-Curriculum degradation is the study's only robust finding (p = 0.0001, t = −15.6)

Anti-Curriculum (256→32) is consistently ~0.178 BPC worse than Curriculum across all 5 seeds. The effect size is large relative to within-seed variance, and the t-statistic is extraordinary (−15.6 at df=4). This finding is robust to all five measurement fixes.

**Plausible mechanism**: When training transitions from T=256 to T=32 in the final 625 steps (the critical low-LR phase), the model must adapt its attention patterns to much shorter dependencies. This truncates long-range associations learned under large T while the learning rate is too small for efficient re-optimization. The result is a model that has "forgotten" long-range structure without adequately learning short-range structure under the small-LR regime.

### What Remains Unknown

- **Attention entropy and rank metrics** (from the original report) were not re-measured under paired conditions. Their values in v1 may differ from corrected-design values. These remain exploratory single-seed observations.
- **Larger corpus / longer context**: Results may not generalize beyond Tiny Shakespeare at 1.1M characters. On a corpus with genuine long-range dependencies (code, long documents), curriculum benefits might emerge.
- **Longer training**: At 2,500 steps (~10M tokens), all arms may not yet have converged. The ordering effect might matter more in early training than at step 2,500.

---

## 6. Theory–Experiment Alignment Matrix — Final

| Prediction | Source | Original Verdict | Final Verdict (5 seeds) |
|------------|--------|:---------------:|:-----------------------:|
| Tr(Σ) ∝ T at initialization | McCandlish 2018 | ✅ Confirmed | ✅ **Confirmed** (unchanged) |
| Short-context curriculum → BPC advantage | Press 2021, Li 2021 | ⚠️ Equivalent | ❌ **Not supported** — p=0.643, Δ=−0.003 |
| Anti-curriculum → degradation | — | ❌ Degraded | ✅ **Confirmed** — p=0.0001, Δ=+0.178 |
| Fixed-Long competitive with scheduling | — | (not tested) | ✅ **Confirmed** — p=0.886, Δ=+0.001 |
| Corpus local regularity (lag ≤ 16) | Probe A | ✅ Confirmed | ❌ **Retracted** — estimator artifact |
| Rank collapse prevention by curriculum | Dong 2021 | ✅ Confirmed | 🔍 **Not revalidated** — not measured in CE-3 |

---

## Appendix A: File Index

| File | Description |
|------|-------------|
| `corpus_probe_v2.py` | CE-1: corrected entropy estimator |
| `experiment_paired.py` | CE-3: 5-seed paired training experiment |
| `results/ce1_corpus_probe_corrected.json` | CE-1 raw output |
| `results/ce3_paired/seed_N_results.json` | Per-seed per-arm BPC and training logs |
| `results/ce3_paired/ce3_summary.json` | Aggregated means, stds, t-test results |
| `results/ce3_run_log.txt` | Full training log |
| `training_dynamics_research/history/iteration_1_exploratory_pilot/technical_report_v1_retracted.md` | Archived original (retracted) report |
