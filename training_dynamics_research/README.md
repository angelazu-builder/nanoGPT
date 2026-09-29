# Training Dynamics Research: Reading Map

This directory records how an initially broad curriculum-learning question became a preregistered recovery study. Read it as a research process, not as several competing “final” reports.

## Research journey

| Iteration | Question | What changed | Defensible outcome |
|---|---|---|---|
| 1. Exploratory pilot | Does context-length order matter? | Single-seed schedules plus diagnostic probes | A dramatic descending deficit appeared, but measurement problems made it a clue rather than a conclusion. |
| 2. Paired pilot | Does the deficit survive corrected measurement? | Five paired seeds, shared initialization and data manifests, fixed held-out targets | The descending terminal deficit replicated; curriculum showed no detectable advantage over shuffled or fixed-long. Terminal context remained confounded with order. |
| 3. Recovery study | Is the deficit persistent under a common long-context condition? | Matched block permutations, common `T=256` recovery, multi-horizon process and anchor panels, preregistered decisions | **Large deficit reversed; residual sign reversal unresolved.** |

The conclusion is that **the large effect of training order did not persist. Final performance was driven mainly by the most recent training context, although a smaller ordering effect cannot yet be ruled out.**

## Final deliverables

- [Concise technical report](recovery_study/TECHNICAL_REPORT_CONCISE_final.md) — 2–4 page account of the question, controlled test, core result, revised interpretation, and remaining unknowns.
- [Final technical report](recovery_study/TECHNICAL_REPORT_final.md) — full process, audit trail, failures, registered analysis, results, and remaining unknowns.
- [Final paper source](paper_icml2026/paper_final.tex) — named ICML-style preprint.
- [Final paper PDF](../output/pdf/context_order_recovery_icml2026_final.pdf).
- [Frozen preregistration](recovery_study/PREREGISTRATION.md).

Only final deliverables use the `_final` suffix. Executable modules and machine-generated artifacts retain stable names so imports and reproduction commands do not break.

## Iteration records and figures

### 1. Exploratory pilot

These records preserve the initial single-seed experiment and the measurement problems discovered afterward. They are historical exploratory evidence, not current causal evidence.

- [Reconstructed design and controls](history/iteration_1_exploratory_pilot/PREREGISTRATION_RECONSTRUCTED.md) — retrospective, not a prospective preregistration.
- [Original theory/computation/experiment proposal](history/iteration_1_exploratory_pilot/plan_proposal_original.md).
- [Original technical report with retracted claims](history/iteration_1_exploratory_pilot/technical_report_v1_retracted.md).
- [Theory–computation–experiment dashboard](../results/theory_computation_experiment_dashboard.png).
- [Training dynamics comparison](../results/curriculum_dynamics_comparison.png).

### 2. Paired pilot

Relative to Iteration 1, this study added five paired seeds, shared initialization, context-specific data manifests, held-out validation text, fixed target positions, and corrected same-target evaluation. These changes made the terminal deficit reproducible rather than anecdotal, but training order remained confounded with the final context seen by each model.

- [Reconstructed design, pairing, and controls](history/iteration_2_paired_pilot/PREREGISTRATION_RECONSTRUCTED.md) — retrospective, not a prospective preregistration.
- [Corrected paired-pilot technical report](history/iteration_2_paired_pilot/technical_report_v3_paired_pilot.md).
- [Legacy summary retained for audit](history/iteration_2_paired_pilot/technical_report_v2_legacy.md) — contains language later judged too strong.
- [Five-seed corrected training-dynamics dashboard](../results/ce3_corrected_dashboard.png).

### 3. Recovery study

Relative to Iteration 2, this study added matched block permutations followed by a common `T=256` recovery stage, plus four-horizon process and anchor panels measured across 17 checkpoints. Preregistered estimands, thresholds, decision rules, and figures turned the terminal comparison into a direct test of whether the deficit persisted after recent training context was matched.

- [Recovery-study implementation and figure gallery](recovery_study/README.md#preregistered-figure-gallery).
- [Prospective preregistration](recovery_study/PREREGISTRATION.md) — setup, held constants, estimands, and decision rules frozen before formal training.
- [Preregistered figure amendment](recovery_study/PREREGISTRATION_AMENDMENT_001_FIGURES.md) — frozen before formal results were produced or inspected.
- [Formal artifacts, tables, and figures](../results/recovery_study/).
- [Central preregistered recovery trajectory](../results/recovery_study/figures/main_02_recovery_gap_by_horizon.png).

## Why the documents have different roles

- The **history** preserves earlier beliefs and errors; it is not current guidance.
- The **preregistration** freezes estimands and decision rules before results.
- The **technical report** preserves the audit trail, failures, corrections, complete evidence, and unknowns.
- The **paper** compresses the strongest defensible contribution for an external reader.
- The **methodology notes and skills** generalize what was learned about ablations, evidence priority, scientific tradeoffs, and code quality.
