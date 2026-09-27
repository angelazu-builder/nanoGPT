# Context-Length Order Recovery Study

## 1. Project decision

Do not immediately run a large three-layer research program. On an Apple M3 with 24GB memory, use a staged, compute-aware design:

1. reanalyze existing results at negligible training cost;
2. run one decisive `3 arms × 3 paired seeds` recovery experiment;
3. include context-sensitive evaluation in the same experiment;
4. run an optimizer-state mechanism experiment only if a persistent gap survives recovery.

## 2. Research question

> Does context-length block order create persistent path dependence after all models receive the same final long-context training, or is the observed anti-curriculum degradation a reversible terminal-context effect?

The study is intentionally narrow. It does not attempt to prove that curriculum learning generally works or fails.

## 3. Hypotheses and estimands

At step 2000, immediately before common recovery:

\[
\Delta_{pre}
=
\mathrm{BPC}_{descending,2000}
-
\mathrm{BPC}_{ascending,2000}.
\]

After 500 common `T=256` recovery steps:

\[
\Delta_{post}
=
\mathrm{BPC}_{descending,2500}
-
\mathrm{BPC}_{ascending,2500}.
\]

If \(|\Delta_{pre}|\ge 0.02\), define:

\[
R=\frac{\Delta_{post}}{\Delta_{pre}}.
\]

### Primary hypothesis

The original degradation is mainly terminal-context mismatch:

\[
R<0.25.
\]

That is, common long-context recovery removes at least 75% of the gap.

### Persistent-effect criterion

Proceed to a mechanism study only if:

- mean \(\Delta_{post}>0.03\) BPC;
- all 3 paired seeds have the same sign;
- no run failed validity checks.

The 0.03 BPC threshold is a project-specific practical threshold, not a universal standard.

## 4. Experimental arms

Use seeds `42, 43, 44`.

Each run contains 2,500 updates and 4,096 tokens/update. The first 2,000 steps contain four 500-step blocks. The last 500 steps are common recovery.

| Arm | Steps 1–2000 | Steps 2001–2500 |
|---|---|---|
| ascending | 32→64→128→256 | 256 |
| descending | 256→128→64→32 | 256 |
| nonmonotonic | 64→256→32→128 | 256 |

The fixed non-monotonic permutation is used for all seeds so the treatment definition does not change by seed.

Optional secondary baseline:

| Arm | Full training |
|---|---|
| fixed-long | T=256 for 2,500 steps |

Fixed-long is an engineering baseline, not part of the pure order estimand because its context histogram differs.

## 5. Controlled variables

Hold fixed across primary arms:

- model architecture and parameter count;
- tokenizer, vocabulary, and dataset split;
- seed-specific initialization;
- AdamW configuration and gradient clipping;
- global-step learning-rate schedule;
- 4,096 tokens/update;
- 500 exposures to each context before recovery;
- transition count and block length;
- data manifest indexed by context occurrence;
- validation target positions;
- evaluation schedule;
- final 500-step `T=256` recovery.

The intended treatment is only the permutation of the four pre-recovery blocks.

## 6. Code layout

Create an isolated implementation rather than modifying the historical experiment in place:

```text
training_dynamics_research/recovery_study/
├── config.py
├── schedules.py
├── manifests.py
├── evaluation.py
├── runner.py
├── analyze.py
└── README.md
```

Results:

```text
results/recovery_study/
├── run_config.json
├── validation_manifest.npz
├── seed_42/{ascending,descending,nonmonotonic}/
├── seed_43/{ascending,descending,nonmonotonic}/
├── seed_44/{ascending,descending,nonmonotonic}/
└── summary.json
```

### Schedule builder

```python
SCHEDULES = {
    "ascending": [32, 64, 128, 256],
    "descending": [256, 128, 64, 32],
    "nonmonotonic": [64, 256, 32, 128],
}

def build_schedule(name, block_steps=500, recovery_steps=500):
    result = []
    for context_length in SCHEDULES[name]:
        result.extend([context_length] * block_steps)
    result.extend([256] * recovery_steps)
    assert len(result) == 2500
    return result
```

### Manifest design

Within each seed, all arms use the same draws for the same context-occurrence index. Required occurrences are:

```text
T=32:   500 steps × batch 128
T=64:   500 steps × batch 64
T=128:  500 steps × batch 32
T=256: 1000 steps × batch 16
```

Maintain a separate occurrence counter per context. The first 500 `T=256` occurrences serve the scheduled block; occurrences 500–999 serve common recovery.

Sampling with `randint` is permitted, but document it as sampling with replacement and potentially overlapping windows.

### Required tests

Before training, verify:

1. every schedule is exactly 2,500 steps;
2. each pre-recovery context appears exactly 500 times;
3. the final 500 steps are all `T=256`;
4. every step satisfies `B×T=4096`;
5. manifest indices are in bounds;
6. within a seed, arms receive identical draws for the same context occurrence;
7. different seeds do not have identical manifests;
8. checkpoints reload and reproduce evaluation output.

## 7. Evaluation

### Fixed validation manifest

Create and save one validation manifest, preferably `64 × 256` tokens. Reuse it for every arm, seed, checkpoint and context horizon. Use microbatches if memory requires it.

### Evaluation times

Evaluate every 100 steps and around transitions:

```text
499, 500, 501, 510, 550
999, 1000, 1001, 1010, 1050
1499, 1500, 1501, 1510, 1550
1999, 2000, 2001, 2010, 2050
2100, 2200, 2300, 2400, 2500
```

Save full model checkpoints at steps 2000 and 2500. Intermediate points need metrics only.

### Primary metric

Fixed-target validation BPC evaluated with `T=256` at steps 2000 and 2500.

### Secondary metrics

- validation BPC AUC for steps 1–2000, 2000–2500, and full training;
- transition-local validation loss shock;
- wall-clock, seconds/step, and peak memory;
- time/tokens to a prespecified BPC threshold, if all arms reach it.

### Context intervention

At checkpoints 2000 and 2500, score identical targets with contexts `32, 64, 128, 256`.

For each target token:

\[
C_i=\ell_i(32)-\ell_i(256),
\qquad
\ell_i(T)=-\log_2P(x_i\mid c_T).
\]

Report:

- mean and median \(C_i\);
- fraction with \(C_i>0.1\) bits;
- BPC on the top 10% most context-sensitive targets;
- paired ascending–descending difference on those same targets.

## 8. Runtime discipline on Apple M3 24GB

1. Run a reduced smoke test with 10 steps/block, 10 recovery steps, one seed and all arms.
2. Run one 100-step benchmark before estimating total runtime.
3. Synchronize MPS before and after timing with `torch.mps.synchronize()`.
4. Execute formal runs sequentially to avoid memory pressure.
5. Rotate arm order across seeds:

```text
seed 42: ascending → descending → nonmonotonic
seed 43: descending → nonmonotonic → ascending
seed 44: nonmonotonic → ascending → descending
```

6. Complete every preregistered arm before interpreting the result.

## 9. Statistical reporting

With three seeds, emphasize estimation rather than significance testing. Report:

- all raw seed values;
- paired differences;
- paired mean and standard deviation;
- 95% paired t interval, explicitly noting `df=2` and high uncertainty;
- direction consistency;
- \(\Delta_{pre}\), \(\Delta_{post}\), recovery ratio and fraction recovered.

Do not claim equivalence from a non-significant p-value. Three seeds are intended to detect or reject a large practical effect quickly, not prove a small effect absent.

## 10. Stop/go decisions

### A. Pre-recovery gap does not replicate

If \(|\Delta_{pre}|<0.02\): stop. Conclude that the original large degradation is not robust to the matched-block design.

### B. Recovery removes at least 75%

If \(R<0.25\): stop. Conclude that the degradation is predominantly reversible terminal-context/recency effect, not strong evidence of persistent path dependence.

### C. Partial persistence

If `0.25 ≤ R < 0.75`, describe temporary path dependence. Continue only if all seeds agree in direction and the remaining effect is practically meaningful.

### D. Persistent effect

If mean \(\Delta_{post}>0.03\) and all three seeds agree, run a minimal mechanism screen:

```text
order: ascending vs descending
optimizer treatment: preserve vs reset Adam moments at transitions
```

Run one exploratory seed first. Expand to three only if resetting moments materially changes the order gap.

## 11. Technical report structure

1. **Abstract** — question, confound, matched-recovery design, quantitative answer, bounded conclusion.
2. **Research question** — define persistent path dependence and the estimands.
3. **Prior evidence and hypotheses** — separate literature, prior project observation, and preregistered prediction.
4. **Methods** — model/data, schedules, pairing, manifest semantics, metrics, compute and statistical policy.
5. **Results** — pre-recovery replication, post-recovery gap, recovery trajectory, context-sensitive tokens, compute cost.
6. **Failed hypotheses and contradictions** — mandatory section; state where predictions failed or measurements disagreed.
7. **Interpretation** — distinguish reversible recency, temporary path dependence, and persistent order effect.
8. **Limitations** — three seeds, one small corpus/model/tokenizer, context ≤256, overlapping windows, project-specific thresholds.
9. **Conclusion** — answer four questions: Did the gap replicate? How much recovered? Is mechanism work justified? What remains unknown?

## 12. Permitted conclusion templates

If recovery removes the gap:

> Descending context order produced a reversible terminal-context effect. After matched long-context recovery, we found no compelling evidence of persistent path dependence at this scale.

If a gap remains:

> A persistent order-dependent difference remained after matched long-context recovery. This supports, but does not yet explain, training path dependence and motivates a targeted optimizer-state intervention.

Never write “the model forgot long-range structure” unless token-level or representation evidence directly supports that mechanism.

## 13. Deliverables

- frozen preregistration and any timestamped amendments;
- isolated experiment implementation and tests;
- frozen run config and validation manifest;
- nine complete primary run directories;
- raw per-seed metrics and summary JSON;
- learning/recovery curve and context-sensitive-token plot;
- technical report with failures and unknowns;
- exact reproduction commands.
