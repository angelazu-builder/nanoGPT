# 04 — Context-Length Order Recovery Study

## Status

This document is the execution specification for the next experiment.

- Hardware constraint: Apple M3, 24GB unified memory
- Formal design: **3 arms × 3 paired seeds = 9 runs**
- Primary purpose: distinguish reversible terminal-context/recency effects from persistent context-order path dependence
- Statistical posture: estimation-first; this experiment is not powered to prove small-effect equivalence

Before formal training, copy the frozen design and decision rules into `training_dynamics_research/recovery_study/PREREGISTRATION.md`, commit it, and record the commit hash in `logbook.md`.

## 1. Research question

> After matching context exposure, transition structure, training budget, validation targets, and a final long-context recovery stage, does the temporal order of context-length blocks leave a persistent effect on language-model performance?

The previous experiment observed that descending context order performed about 0.178 BPC worse than ascending curriculum. However, descending training ended at `T=32`, ascending training ended at `T=256`, and evaluation primarily used `T=256`. The previous result therefore combined:

1. historical block-order effects;
2. recency from the final training context;
3. final-training/evaluation context mismatch;
4. possible interactions between context length and the late low-learning-rate phase.

The recovery study does not assume that recency disappears after a fixed number of steps. It measures how the paired performance gap changes while all arms receive identical `T=256` recovery training.

## 2. Hypotheses and estimands

For recovery step \(k\), define:

\[
\Delta(k)
=
\mathrm{BPC}_{descending,\,2000+k}
-
\mathrm{BPC}_{ascending,\,2000+k}.
\]

The main checkpoints are:

\[
\Delta_{pre}=\Delta(0)
\]

and:

\[
\Delta_{post}=\Delta(500).
\]

Define absolute recovery:

\[
G_{recovered}=\Delta(0)-\Delta(500).
\]

Only when:

\[
|\Delta(0)|\ge 0.02\ \mathrm{BPC}
\]

may the recovery ratio be interpreted:

\[
R=\frac{\Delta(500)}{\Delta(0)}.
\]

### Primary hypothesis

The original descending degradation is mainly a reversible terminal-context/recency effect:

\[
R<0.25.
\]

This means at least 75% of the pre-recovery gap is removed during matched long-context recovery.

### Persistent-effect candidate

Evidence is considered sufficient to justify a subsequent mechanism experiment only if all conditions hold:

1. mean \(\Delta(500)>0.03\) BPC;
2. all three paired seeds have \(\Delta(500)>0\);
3. the mean gap has approximately plateaued:

   \[
   |\Delta(500)-\Delta(400)|<0.01\ \mathrm{BPC};
   \]

4. no run failed validity or reproducibility checks.

This is evidence for a **persistent-effect candidate**, not proof of a permanent or unique mechanism.

## 3. Meaning of the thresholds

The thresholds are project-specific decision rules and must be frozen before training.

### 0.02 BPC: minimum interpretable pre-gap

If \(|\Delta(0)|<0.02\), the denominator of the recovery ratio is too small for stable interpretation. The correct conclusion is that the previous large degradation did not clearly replicate under the matched-block design.

### 0.25: recovery-ratio criterion

`R < 0.25` means at least 75% of the measured pre-recovery gap was removed. It is a directional decision rule, not a universal scientific constant.

### 0.03 BPC: additional-compute threshold

\[
2^{0.03}\approx1.021.
\]

Thus 0.03 BPC corresponds to roughly a 2.1% perplexity ratio. It is the minimum residual effect considered large enough to justify spending additional compute on a mechanism study. It is not a significance boundary or a threshold derived from prior literature.

## 4. Formal experimental arms

Use paired seeds:

```python
SEEDS = [42, 43, 44]
```

Each run contains 2,500 optimizer steps and 4,096 target tokens per update.

| Arm | Steps 1–2000 | Steps 2001–2500 | Role |
|---|---|---|---|
| `ascending` | 32→64→128→256 | `T=256` recovery | Primary |
| `descending` | 256→128→64→32 | `T=256` recovery | Primary |
| `nonmonotonic` | 64→256→32→128 | `T=256` recovery | Exploratory control |

Each pre-recovery block lasts exactly 500 steps.

### Why retain the nonmonotonic arm

`64→256→32→128` is one fixed, prespecified non-monotonic permutation. It:

- exposes every context exactly once before recovery;
- uses the same 500-step block length;
- has the same three pre-recovery block boundaries;
- contains both increasing and decreasing transitions;
- ends pre-recovery at an intermediate context length.

It does **not** represent the population of all non-monotonic schedules. It is exploratory and is excluded from the primary hypothesis test. Its purpose is to show whether the observed trajectory appears specific to the two monotonic extremes.

Do not describe this arm as a bootstrap or randomized-order estimate.

### Fixed-Long status

Do not rerun Fixed-Long in this study. Existing Fixed-Long results may be shown as a historical engineering reference, clearly labeled as not belonging to the new matched primary comparison.

## 5. What is held constant

Across the three formal arms, hold constant:

- model architecture, parameter count, RoPE implementation, tokenizer, and vocabulary;
- dataset file, encoding, and train/validation split;
- seed-specific initial model weights;
- AdamW configuration, weight decay, epsilon, beta values, and clipping threshold;
- optimizer-state initialization and the rule that optimizer state is not reset at transitions;
- global-step learning-rate schedule;
- 2,500 optimizer steps;
- 4,096 target tokens per update;
- 500 scheduled steps at each of `T=32,64,128,256`;
- 500 final recovery steps at `T=256`;
- pre-recovery block length and block-boundary count;
- context-specific scheduled data draws;
- recovery data draws, matched exactly by recovery step;
- validation examples, target positions, scoring code, and evaluation checkpoints;
- software environment, code commit, and device settings.

The intended treatment is:

\[
\boxed{\text{the temporal permutation of the four pre-recovery context blocks}}.
\]

Because context blocks occur at different global steps, the treatment necessarily includes their interaction with model state, optimizer history, and the fixed global learning-rate trajectory. The estimand is therefore the total block-order effect under this training policy, not an abstract order effect independent of learning rate.

### Remaining transition asymmetry

All arms have the same number of scheduled block boundaries, but entering recovery is not the same actual context switch:

- ascending: `256→256`;
- descending: `32→256`;
- nonmonotonic: `128→256`.

This is intentional. The recovery curve measures how effects associated with the immediately preceding context decay under a common target task.

## 6. Tokens per update

Hold:

\[
B\times T=4096.
\]

| Context \(T\) | Batch size \(B\) |
|---:|---:|
| 32 | 128 |
| 64 | 64 |
| 128 | 32 |
| 256 | 16 |

Every arm sees:

\[
2500\times4096=10{,}240{,}000
\]

target tokens.

Token matching is not compute matching. Record measured wall-clock time, seconds per step, and peak MPS memory.

## 7. Code layout

Implement the study separately from the historical experiment:

```text
training_dynamics_research/recovery_study/
├── PREREGISTRATION.md
├── config.py
├── schedules.py
├── manifests.py
├── evaluation.py
├── runner.py
├── analyze.py
└── README.md
```

Store outputs under:

```text
results/recovery_study/
├── run_config.json
├── validation_manifest.npz
├── seed_42/{ascending,descending,nonmonotonic}/
├── seed_43/{ascending,descending,nonmonotonic}/
├── seed_44/{ascending,descending,nonmonotonic}/
└── summary.json
```

## 8. Schedule implementation

```python
SEEDS = [42, 43, 44]

SCHEDULES = {
    "ascending": [32, 64, 128, 256],
    "descending": [256, 128, 64, 32],
    "nonmonotonic": [64, 256, 32, 128],
}

BLOCK_STEPS = 500
RECOVERY_STEPS = 500
MAX_STEPS = 2500


def build_scheduled_contexts(arm):
    contexts = []
    for context_length in SCHEDULES[arm]:
        contexts.extend([context_length] * BLOCK_STEPS)
    assert len(contexts) == 2000
    return contexts
```

Recovery is implemented separately and must not be appended through the scheduled-context occurrence counters.

## 9. Manifest design

Use two explicit namespaces:

```text
scheduled_manifest
recovery_manifest
```

### Scheduled manifest

For each seed and context length:

```text
T=32:  500 × 128 draws
T=64:  500 × 64 draws
T=128: 500 × 32 draws
T=256: 500 × 16 draws
```

```python
def generate_scheduled_manifest(seed, train_length):
    manifest = {}

    for T in [32, 64, 128, 256]:
        B = 4096 // T
        rng = np.random.RandomState(seed * 10_000 + T)
        manifest[T] = rng.randint(
            0,
            train_length - T - 1,
            size=(500, B),
        )

    return manifest
```

Within a seed, the \(k\)-th scheduled occurrence of a given context uses the same draws in all arms, regardless of its global step.

### Recovery manifest

```python
def generate_recovery_manifest(seed, train_length):
    rng = np.random.RandomState(seed * 10_000 + 9_999)
    return rng.randint(
        0,
        train_length - 256 - 1,
        size=(500, 16),
    )
```

At recovery step \(k\), every arm within a seed uses exactly `recovery_manifest[k]`.

Sampling uses replacement and text windows may overlap. Do not describe samples as unique or non-overlapping.

### Required manifest assertions

Tests must verify scheduled-manifest equality across arms for each context and recovery-manifest equality across arms. Also verify index bounds, array shapes, and distinct manifests across seeds.

## 10. Validation design

Create one fixed validation manifest:

```python
N_VAL_SEQUENCES = 64
VAL_CONTEXT = 256
```

This provides four times as many evaluation sequences as the previous 16-sequence setup while remaining inexpensive on M3. It does not justify a prespecified claim that effects of ±0.01 BPC are detectable.

All arms, seeds, checkpoints, and context horizons use identical validation endpoints. Microbatching may change for memory reasons, but targets and scoring must not change.

Store loss separately for each validation sequence. Estimate evaluation uncertainty by resampling sequences as blocks. Do not treat individual tokens in the same sequence as independent bootstrap samples.

## 11. Evaluation checkpoints

Routine validation occurs every 100 global steps. Add recovery evaluations at:

```python
RECOVERY_EVAL_STEPS = [0, 10, 25, 50, 100, 200, 300, 400, 500]
```

These correspond to global steps:

```text
2000, 2010, 2025, 2050, 2100, 2200, 2300, 2400, 2500
```

Save full model checkpoints at global steps 2000 and 2500. Save metrics, but not necessarily model weights, at intermediate recovery points.

## 12. Metrics

### Confirmatory outcomes

- \(\Delta(0)\);
- \(\Delta(500)\);
- absolute gap recovered;
- recovery fraction or ratio when \(|\Delta(0)|\ge0.02\);
- direction of paired differences across seeds.

### Secondary outcomes

- validation BPC AUC over steps 1–2000;
- recovery BPC AUC over steps 2000–2500;
- transition-local validation loss shock;
- wall-clock time, seconds/step, and peak memory;
- time or tokens to a prespecified BPC threshold, if all arms reach it.

### Exploratory outcomes

- nonmonotonic comparisons;
- descriptive recovery-curve fits;
- attention entropy, effective rank, or optimizer diagnostics;
- context-sensitive token analysis.

Exploratory outcomes cannot retroactively redefine the primary hypothesis.

## 13. Same-target context intervention

At global steps 2000 and 2500, score identical target positions with context horizons `32, 64, 128, 256`.

For target token \(i\), compute:

\[
\ell_i(T)=-\log_2P(x_i\mid c_T)
\]

and:

\[
C_i=\ell_i(32)-\ell_i(256).
\]

Report mean and median \(C_i\), the fraction with \(C_i>0.1\) bits, BPC on the top 10% most context-sensitive targets, and the paired ascending–descending difference on the same targets.

The target subset must be defined without selecting whichever subset maximizes the reported arm difference. Prefer a treatment-blind aggregate or an external/reference model.

## 14. Statistical reporting

With three seeds, emphasize estimation rather than significance testing. Report:

- all raw seed-level values;
- all paired differences;
- paired mean and paired standard deviation;
- a 95% paired t interval, explicitly marked as `df=2` and highly uncertain;
- direction consistency: `3/3`, `2/3`, or `1/3`;
- the full recovery trajectory.

Do not infer equivalence from `p>0.05`. Permitted wording is:

> No difference was detected with three paired seeds; effects below the study's resolution remain uncertain.

Do not write “same,” “equivalent,” or “no effect” unless a separately powered equivalence design is completed.

## 15. Nonmonotonic analysis policy

The primary confirmatory contrast is always descending minus ascending.

Exploratory contrasts may include:

\[
\Delta_{N-A}(k)
=
\mathrm{BPC}_{nonmonotonic}(k)
-
\mathrm{BPC}_{ascending}(k)
\]

and:

\[
\Delta_{D-N}(k)
=
\mathrm{BPC}_{descending}(k)
-
\mathrm{BPC}_{nonmonotonic}(k).
\]

Rules:

- show all raw seeds and trajectories;
- do not generalize from one permutation to non-monotonic schedules as a class;
- do not change the primary hypothesis based on this arm;
- do not use this arm alone to establish a mechanism.

## 16. Stop/go rules

### Case A: pre-recovery gap does not replicate

If \(|\mathrm{mean}\ \Delta(0)|<0.02\), stop mechanism work and do not interpret the recovery ratio.

### Case B: gap is mostly removed

If \(|\mathrm{mean}\ \Delta(500)|\le0.02\), or interpretable \(R<0.25\), conclude that the result primarily supports a reversible terminal-context/recency explanation. Stop mechanism expansion.

### Case C: result is unresolved

If the remaining mean gap is between 0.02 and 0.03 BPC, or seed directions disagree, report the result as unresolved. Do not claim equivalence or a persistent effect.

### Case D: gap remains large but is still recovering

If mean \(\Delta(500)>0.03\) and mean \(\Delta(400)-\Delta(500)>0.01\), the 500-step recovery is insufficient. Extend **all three arms for all three seeds** by another 500 matched `T=256` steps, using a separately pregenerated `extended_recovery_manifest` shared by recovery step.

### Case E: persistent-effect candidate

Proceed to a targeted optimizer-state experiment only if:

- mean \(\Delta(500)>0.03\);
- `3/3` paired seeds have positive \(\Delta(500)\);
- \(|\Delta(500)-\Delta(400)|<0.01\);
- all validity checks pass.

The next experiment should initially test only:

```text
order: ascending vs descending
optimizer treatment: preserve vs reset Adam moments at transitions
```

Run one exploratory seed first; expand only if the intervention materially changes the gap.

## 17. Runtime procedure on Apple M3 24GB

1. Run schedule and manifest unit tests.
2. Run a smoke test with 10 steps/block, 10 recovery steps, one seed, and all arms.
3. Confirm checkpoint reload and identical validation outputs.
4. Run one 100-step benchmark and record evaluation overhead.
5. Execute formal runs sequentially.
6. Synchronize MPS before and after timing with `torch.mps.synchronize()`.
7. Rotate arm execution order:

```text
seed 42: ascending → descending → nonmonotonic
seed 43: descending → nonmonotonic → ascending
seed 44: nonmonotonic → ascending → descending
```

8. Complete all nine preregistered runs before interpreting results.

## 18. Preregistration requirements

Before formal training, freeze:

- arms and their exact schedules;
- seeds;
- model and optimizer config;
- manifest generation and RNG namespaces;
- validation manifest;
- primary, secondary, and exploratory metrics;
- recovery checkpoints;
- the 0.02, 0.25, 0.03, and 0.01 decision thresholds;
- adaptive-extension rule;
- run exclusion criteria;
- code commit and environment metadata.

Commit `PREREGISTRATION.md` before producing formal results. If a change becomes necessary, add a timestamped amendment explaining whether any formal result had already been observed. Never silently rewrite the frozen specification.

## 19. Technical report structure

1. **Abstract** — question, original confound, matched-recovery design, numerical answer, bounded conclusion.
2. **Research question** — define recency and persistent path dependence operationally.
3. **Prior evidence and hypotheses** — separate literature, previous project observations, and preregistered predictions.
4. **Methods** — arms, held constants, pairing, manifest namespaces, validation, metrics, hardware, and statistical policy.
5. **Results: pre-recovery replication** — report \(\Delta(0)\) before discussing recovery.
6. **Results: recovery trajectory** — report \(\Delta(k)\), \(\Delta(500)\), absolute recovery, and ratio when valid.
7. **Results: exploratory nonmonotonic control** — clearly separated from the primary comparison.
8. **Results: context-sensitive targets and compute cost**.
9. **Failed hypotheses and contradictions** — mandatory, including measurement or design surprises.
10. **Interpretation** — distinguish reversible recency, unresolved residual effects, and persistent-effect candidates.
11. **Limitations** — three seeds, single model/corpus/tokenizer, context ≤256, overlapping windows, and project-specific thresholds.
12. **Conclusion** — state whether the original gap replicated, how much recovered, whether mechanism work is justified, and what remains unknown.

## 20. Permitted conclusion templates

### If the gap disappears

> Descending context order produced a largely reversible terminal-context effect. After matched long-context recovery, we found no compelling evidence of a large persistent path dependence at this scale. Smaller effects remain unresolved with three seeds.

### If the gap remains but recovery is ongoing

> A residual difference remained after 500 recovery steps, but the gap was still decreasing. The current experiment cannot distinguish slow recovery from persistent path dependence.

### If a stable residual remains

> A directionally consistent residual difference remained after matched long-context recovery and appeared stable over the final recovery interval. This is candidate evidence for persistent path dependence and motivates a targeted optimizer-state intervention; it does not yet establish the mechanism.

Never write “the model forgot long-range structure” unless direct token-level or representation evidence supports that claim.

## 21. Deliverables

- frozen preregistration and any timestamped amendments;
- schedule and manifest unit tests;
- isolated recovery-study implementation;
- frozen run config, environment metadata, and validation manifest;
- nine complete formal run directories;
- raw per-seed metrics and paired differences;
- summary JSON;
- recovery-gap plot and learning curves;
- context-sensitive-target analysis;
- wall-clock and memory measurements;
- technical report with explicit failures, limitations, and unknowns;
- exact reproduction commands.
