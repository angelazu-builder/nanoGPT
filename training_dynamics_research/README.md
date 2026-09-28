# Training Dynamics Research: Reading Map

This directory records how an initially broad curriculum-learning question became a preregistered recovery study. Read it as a research process, not as several competing “final” reports.

## Recommended reading order

1. **Current conclusion:** [`recovery_study/TECHNICAL_REPORT_final.md`](recovery_study/TECHNICAL_REPORT_final.md)
2. **Frozen design:** [`recovery_study/PREREGISTRATION.md`](recovery_study/PREREGISTRATION.md)
3. **Paper:** [`paper_icml2026/paper_final.tex`](paper_icml2026/paper_final.tex) or [`../output/pdf/context_order_recovery_icml2026_final.pdf`](../output/pdf/context_order_recovery_icml2026_final.pdf)
4. **Earlier iterations:** `history/`, when you need to see why the final design changed.
5. **Reusable methodology:** `research_methodology/`.

## Iteration 1 — Exploratory pilot

**Question.** Could an ascending context-length curriculum change optimization and final performance?

**What happened.** A single-seed experiment produced dramatic diagnostics and a poor descending endpoint. Several claims were later weakened or retracted because targets were not consistently aligned, the corpus probe suffered resubstitution bias, batches were not paired, and exploratory diagnostics were overinterpreted as mechanism evidence.

**Historical records.**

- [`history/iteration_1_exploratory_pilot/plan_proposal_original.md`](history/iteration_1_exploratory_pilot/plan_proposal_original.md) — original theory/computation/experiment proposal.
- [`history/iteration_1_exploratory_pilot/technical_report_v1_retracted.md`](history/iteration_1_exploratory_pilot/technical_report_v1_retracted.md) — original report, retained with its retracted claims for audit.

## Iteration 2 — Corrected paired pilot

**Question.** Does the terminal difference remain after paired data and corrected measurement?

**What changed.** Five seeds, matched initialization, shared context-specific data manifests, held-out validation text, and fixed final target positions.

**What survived.** Curriculum had no detectable advantage over shuffled or fixed-long training. Descending/anti-curriculum ended about `0.178 BPC` worse at `T=256` in all five seeds. This was a stable terminal observation, not a clean ordering mechanism: the final training context, evaluation context, learning-rate phase, and optimizer history remained entangled.

**Historical records.**

- [`history/iteration_2_paired_pilot/technical_report_v3_paired_pilot.md`](history/iteration_2_paired_pilot/technical_report_v3_paired_pilot.md) — corrected full analysis.
- [`history/iteration_2_paired_pilot/technical_report_v2_legacy.md`](history/iteration_2_paired_pilot/technical_report_v2_legacy.md) — legacy summary retained for audit; it contains language later judged too strong.

## Iteration 3 — Preregistered recovery study

**Question.** Does the large positive descending deficit persist when all arms enter the same long-context training condition?

**Design.** Three matched block permutations × three paired seeds, followed by 500 common `T=256` recovery steps. A nested 16-sequence process panel measures four evaluation horizons at 17 checkpoints; a 32-sequence anchor panel supplies the registered endpoints.

**Result.** The pre-recovery `T=256` gap was `+0.7570 BPC`; after recovery it was `−0.0485 BPC`. The large positive deficit rapidly reversed, but the residual exceeded the preregistered practical-removal band and remained uncertain with three seeds.

**Current conclusion.** **Large deficit reversed; residual sign reversal unresolved.**

**Implementation and evidence.**

- `recovery_study/` — executable package, preregistration, final technical report, and tests.
- `../results/recovery_study/` — formal run artifacts, tidy tables, registered figures, and versioned paper figures.
- `paper_icml2026/` — final manuscript source and bibliography.

## Why the documents have different roles

- The **history** preserves earlier beliefs and errors; it is not current guidance.
- The **preregistration** freezes estimands and decision rules before results.
- The **technical report** preserves the audit trail, failures, corrections, complete evidence, and unknowns.
- The **paper** compresses the strongest defensible contribution for an external reader.
- The **methodology notes and skills** generalize what was learned about ablations, evidence priority, scientific tradeoffs, and code quality.
