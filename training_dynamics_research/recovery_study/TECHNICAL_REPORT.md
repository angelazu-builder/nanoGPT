# Context-Length Order Recovery Study

## Technical report

> **Status:** Complete 3-arm × 3-seed controlled study; preregistered endpoint is unresolved  
> **Experiment date:** September 27, 2026  
> **Model:** 4.8M-parameter character-level Transformer on Tiny Shakespeare  
> **Primary metric:** Paired descending-minus-ascending validation BPC at evaluation horizon `T=256`

## Abstract

We tested whether the large long-context deficit previously observed after descending context-length training reflects persistent path dependence or a reversible terminal-context effect. Three schedules—ascending (`32→64→128→256`), descending (`256→128→64→32`), and one prespecified nonmonotonic permutation (`64→256→32→128`)—received matched context exposure for 2,000 steps, followed by 500 identical `T=256` recovery steps. Data draws, initialization within seed, model, optimizer, token throughput, evaluation targets, recovery learning rate, and training budget were controlled.

Before recovery, the paired descending-minus-ascending anchor gap at `T=256` was `+0.7570 BPC` (95% t interval `[+0.3290, +1.1851]`; 3/3 seeds positive). After recovery, it was `−0.0485 BPC` (95% t interval `[−0.1491, +0.0521]`; 0/3 positive). The registered recovery ratio was `R=−0.064`, but the absolute residual `0.0485 BPC` exceeded the preregistered Case B threshold of `0.02 BPC`. The correct registered outcome is therefore: **large deficit reversed; residual sign reversal unresolved**. The original positive deficit did not persist, but three seeds cannot establish whether the small reversed residual is real or explain its mechanism.

## 1. Research question

After matching context exposure, transition structure, training budget, validation targets, and a final long-context recovery stage, does the temporal order of context-length blocks leave a persistent effect on language-model performance?

The primary contrast was

\[
\Delta^A(k)=\operatorname{BPC}^{A}_{descending,256}(2000+k)
-\operatorname{BPC}^{A}_{ascending,256}(2000+k),
\]

measured on the 32-sequence anchor panel at recovery steps `k=0` and `k=500`.

## 2. Hypotheses and decision rules

The quantitative hypothesis was that common `T=256` recovery would remove at least 75% of the pre-recovery gap:

\[
R=\frac{\Delta^A(500)}{\Delta^A(0)}<0.25.
\]

The ratio was descriptive and could not override the absolute residual rules:

- Case A: `|mean Δᴬ(0)| < 0.02 BPC` — the original gap did not replicate.
- Case B: `|mean Δᴬ(500)| ≤ 0.02 BPC` — practically removed.
- Case C: a small, directionally inconsistent, or boundary residual — unresolved.
- Case D: a positive residual above `0.03 BPC` that was still recovering — extend once.
- Case E: a positive residual above `0.03 BPC`, 3/3 positive, and plateaued — persistent-effect candidate.

The observed large negative residual was not a preregistered Case B outcome. It is reported as a Case C boundary/sign-reversal result rather than redefining Case B after observing the data.

## 3. Controlled experimental design

| Component | Frozen design |
|---|---|
| Arms | ascending, descending, one exploratory nonmonotonic permutation |
| Seeds | 42, 43, 44 |
| Scheduled phase | four 500-step context blocks, steps 1–2000 |
| Recovery phase | common `T=256`, steps 2001–2500 |
| Token throughput | `B×T=4096` tokens per update |
| Optimizer | AdamW with preserved state |
| Recovery learning rate | common schedule ending at `min_lr=1e-4` |
| Process panel | 16 fixed sequences, four horizons, 17 checkpoints |
| Anchor panel | 32 fixed sequences, four horizons, steps 2000 and 2500 |
| Pairing | identical initialization and matched data manifests within seed |
| Execution-order control | Latin-square arm rotation across seeds |

The nonmonotonic arm is exploratory. One permutation cannot support claims about nonmonotonic schedules as a class and does not alter the primary descending-versus-ascending decision.

## 4. Validity and completeness checks

The formal analysis gate verified:

- all 9 runs were present, completed, and not smoke tests;
- all runs reached step 2500;
- every run contained all 17 process checkpoints and all four evaluation horizons;
- every run contained anchor evaluations at steps 2000 and 2500;
- the process panel contained 16 sequence-level BPC values and the anchor panel contained 32;
- the first 16 anchor sequences reconstructed the corresponding process statistic;
- all evaluated values were finite.

The aggregate recorded training time was 8,818.7 seconds (approximately 2.45 hours summed across the nine sequential run records). No training rerun or adaptive extension was triggered by the corrected analysis.

## 5. Primary result

| Endpoint | Seed-level paired differences | Mean ± SD BPC | 95% t interval | Direction |
|---|---|---:|---:|---:|
| `Δᴬ(0)`, step 2000 | `+0.5977, +0.7336, +0.9399` | `+0.7570 ± 0.1723` | `[+0.3290, +1.1851]` | 3/3 positive |
| `Δᴬ(500)`, step 2500 | `−0.0663, −0.0022, −0.0770` | `−0.0485 ± 0.0405` | `[−0.1491, +0.0521]` | 0/3 positive |
| Pre-to-post reduction | `+0.6640, +0.7357, +1.0169` | `+0.8055 ± 0.1865` | `[+0.3422, +1.2689]` | 3/3 positive |

`R=−0.064`, corresponding descriptively to 106.4% recovery because the gap crossed zero. This ratio supports strong recovery but does not establish equivalence. Since `|−0.0485|=0.0485>0.02`, the registered Case B practical-removal criterion was not met.

## 6. Multi-horizon and process evidence

### 6.1 Anchor endpoints

Mean descending-minus-ascending anchor differences were:

| Evaluation horizon | Step 2000 | Step 2500 |
|---:|---:|---:|
| `T=32` | `−0.0667` | `−0.0622` |
| `T=64` | `−0.0424` | `−0.0569` |
| `T=128` | `−0.0258` | `−0.0606` |
| `T=256` | `+0.7570` | `−0.0485` |

The large pre-recovery deficit was specific to evaluation at `T=256`; it was not a broad degradation shared by shorter evaluation horizons. After common recovery, all four mean contrasts were modest and negative, although their three-seed intervals were wide and crossed zero.

### 6.2 Recovery dynamics

On the 16-sequence process panel, the mean `T=256` gap fell from `+0.8097 BPC` at `k=0` to `+0.0719` at `k=10`, crossed zero by `k=50`, and ended at `−0.0195` at `k=500`. The context-alignment contrast

\[
A^P(k)=\Delta_{256}^P(k)-\Delta_{32}^P(k)
\]

also collapsed rapidly, showing that the initial arm difference became much less dependent on evaluation horizon during common long-context recovery.

These trajectories support a terminal-context-sensitive interpretation of the large positive deficit. They do not determine whether the smaller negative residual reflects noise, optimization history, or another source of path dependence.

## 7. Contradiction and corrected interpretation

The initial expectation was that recovery would either leave a positive persistent residual or reduce the gap into the registered `±0.02 BPC` practical-removal region. Instead, the estimate crossed zero and ended outside that region.

An initial post-result analysis revision incorrectly added a signed-negative exception to Case B and described the outcome as “entirely reversible recency bias.” That exception was removed because it was not preregistered and would classify arbitrarily large negative residuals as practical removal.

The corrected interpretation is:

1. the large positive descending deficit replicated before recovery;
2. it did not persist under matched `T=256` recovery;
3. recovery produced a small direction-consistent sign reversal in the point estimates;
4. the reversed residual is larger than the registered equivalence threshold but uncertain with three seeds;
5. therefore the primary endpoint is unresolved, while a large persistent descending deficit is not supported.

## 8. Limitations and remaining unknowns

- Three seeds provide weak precision for effects near `0.05 BPC`; the post-recovery 95% interval spans both moderate negative and small positive effects.
- The study distinguishes recovery behavior, not a unique mechanism. It does not isolate optimizer moments, representation changes, or data memorization.
- Recovery adds both long-context exposure and 500 optimization steps. The design operationalizes recency through matched recovery but does not identify every causal component of recovery.
- The `0.02` and `0.03 BPC` thresholds are project-specific practical thresholds, not universal constants.
- The nonmonotonic evidence comes from one prespecified permutation and remains exploratory.
- The result is restricted to this model scale, character tokenizer, corpus, optimizer, schedule, and token budget.

The central remaining question is whether the approximately `−0.05 BPC` post-recovery contrast replicates with greater precision. The present study does not justify a targeted optimizer-state mechanism experiment because the preregistered positive-persistence condition was not met.

## 9. Conclusion

The experiment rejects the simple claim that descending context order leaves a large positive performance deficit after matched long-context recovery. The original `T=256` deficit was large, horizon-specific, and rapidly reversible. However, recovery did not place the paired mean inside the preregistered practical-removal band; it produced a small sign reversal whose confidence interval crosses zero. The defensible conclusion is therefore **large deficit reversed; residual sign reversal unresolved**, not equivalence, confirmation of a unique recency mechanism, or evidence of persistent positive path dependence.

## 10. Provenance and reproduction

| Stage | Commit |
|---|---|
| Frozen preregistration | `f1bf7b7` |
| Formal training code and run artifacts | `30a126cb7fbee5b5d730aeccc6efa33a18cdf4f8` |
| Corrected analysis code | `82a4f0d5b24d27a6fd66bf3fe1878d6839af0ebd` |
| Registered figure-generation code | `82a4f0d5b24d27a6fd66bf3fe1878d6839af0ebd` |
| Post-results manuscript-figure code | `688d98ad543bbe5a996511247f742f8b12ca17d0` |

The corrected analysis and figures were generated without rerunning the models:

```bash
python3 -m unittest training_dynamics_research.recovery_study.test_study
python3 -m training_dynamics_research.recovery_study.analyze
```

Primary artifacts:

- `results/recovery_study/run_config.json`
- `results/recovery_study/summary.json`
- `results/recovery_study/figures/data/recovery_contrasts.csv`
- `results/recovery_study/figures/main_02_recovery_gap_by_horizon.png`
- `results/recovery_study/figures/appendix_A3_paired_seed_endpoints.png`
- `results/recovery_study/figures/appendix_A4_context_alignment.png`
- `results/recovery_study/figures/appendix_A5_nonmonotonic_exploratory.png`

## 11. Post-results paper visualization layer

The registered figures above remain the audit layer. The manuscript layer was created after inspecting the results and changes no data, estimand, threshold, decision rule, or conclusion. The narrative is deliberately hierarchical: rapid reversal is the primary result; the need to combine process, anchor, and multi-horizon measurements is the methodological contribution; horizon dependence is supporting evidence. The nonmonotonic arm remains exploratory and appears only in the appendix. Each figure is exported as a 300-dpi PNG preview and a vector PDF for manuscript use.

### Main Figure 1 — Common long-context recovery rapidly reverses the large deficit, but the residual is unresolved

**Caption.** Common `T=256` recovery rapidly reverses the descending schedule's long-context deficit, but the experiment does not establish equivalence at the registered endpoint. **(A)** Descending-minus-ascending process-panel BPC at evaluation horizon `T=256` over the common recovery stage; the inset expands the first 50 recovery steps. Pale trajectories are the three paired seeds and the heavy trajectory is their mean. Markers denote measured checkpoints; no smoothing or interpolation is used. Process estimates use the nested 16-sequence panel. **(B)** Registered anchor-panel contrasts before and after recovery, using 32 fixed sequences per arm. Lines connect the same seed and therefore encode pairing, not unmeasured intermediate dynamics; black diamonds are three-seed means and the gray band is the preregistered `±0.02 BPC` practical-removal region. **(C)** Seed-level post-recovery contrasts and their mean with a two-sided 95% Student-t interval across the three paired seeds. Positive values indicate higher BPC for descending than ascending. The mean reverses sign (`−0.0485 BPC`), lies outside the practical-removal band, and its interval crosses zero; the residual sign reversal is therefore unresolved.

Artifacts: `main_figure_1_primary_recovery.{png,pdf}`

### Main Figure 2 — The pre-recovery deficit is specific to long-context evaluation

**Caption.** Horizon-specific descending-minus-ascending anchor effects before and after common recovery. **(A)** Before recovery (`k=0`), the large positive deficit occurs at `T=256`, while shorter-horizon contrasts are small and negative. **(B)** After 500 common `T=256` recovery steps, all four point estimates are modest and negative, and all 95% Student-t intervals cross zero. Circles are paired seed effects (`n=3`), black diamonds are means, and colored bars are two-sided 95% Student-t intervals across seeds. The gray band in panel B is the preregistered `±0.02 BPC` practical-removal region. Panels use independent x-axis ranges to show the much smaller post-recovery effects; comparisons of magnitude should use the numeric axes rather than visual bar length alone.

Artifacts: `main_figure_2_horizon_specificity.{png,pdf}`

### Main Figure 3 — Most evaluation-horizon dependence disappears within 50 recovery steps

**Caption.** Recovery dynamics of the process-panel context-alignment contrast, `Aᴾ(k)=Δᴾ₂₅₆(k)−Δᴾ₃₂(k)`. **(A)** Full 500-step recovery window. **(B)** The first 50 steps, shown on an expanded x-axis. Pale trajectories are paired seeds (`n=3`), heavy trajectories are their means, and markers are measured checkpoints. The contrast falls sharply during the first 10 steps and is approximately `0.05 BPC` by step 50. This supports a terminal-context-sensitive account of the original deficit, but does not isolate optimizer state, representation change, or another unique recovery mechanism.

Artifacts: `main_figure_3_context_alignment.{png,pdf}`

### Appendix Figure A1 — Complete recovery trajectories at all evaluation horizons

**Caption.** Descending-minus-ascending process-panel BPC across all measured recovery checkpoints for `T∈{32,64,128,256}`. Pale trajectories are paired seeds (`n=3`) and heavy trajectories are their means. Panels A–C share a y-axis range; panel D (`T=256`) uses an explicitly labelled independent y-axis because its initial gap is an order of magnitude larger. The shorter-horizon means remain modest throughout recovery, whereas the `T=256` mean crosses zero within 50 steps.

Artifacts: `appendix_figure_A1_all_horizon_trajectories.{png,pdf}`

### Appendix Figure A2 — Exploratory results for one prespecified nonmonotonic permutation

**Caption.** Exploratory process-panel contrasts involving the single prespecified nonmonotonic schedule (`64→256→32→128`) over common recovery at four evaluation horizons. Pink denotes nonmonotonic minus ascending and blue denotes descending minus nonmonotonic; pale trajectories are paired seeds (`n=3`) and heavy trajectories are means. This figure describes one permutation only and must not be generalized to nonmonotonic schedules as a class. It does not enter the primary descending-versus-ascending decision.

Artifacts: `appendix_figure_A2_nonmonotonic_exploratory.{png,pdf}`

The presentation follows useful conventions from prior work: show training dynamics at the time scale of the transition, expose raw runs, define every visual encoding in the caption, and pair point estimates with uncertainty. Relevant examples are [Li et al., *Sequence Length Warmup for Large Language Model Pretraining*](https://arxiv.org/abs/2108.06084), [Power et al., *Grokking*](https://arxiv.org/abs/2201.02177), [Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html), and [Ho et al., *Moving beyond P values: data analysis with estimation graphics*](https://www.nature.com/articles/s41592-019-0470-3).
