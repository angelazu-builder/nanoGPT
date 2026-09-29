---
name: rigorous-ablation-review
description: Audit or design controlled ML ablation studies, especially training-dynamics experiments, by identifying the estimand, changed variables, confounds, measurement validity, uncertainty, mechanism evidence, compute constraints, and warranted conclusion strength. Use when reviewing an experiment, technical report, ablation table, training curriculum, or proposed mechanism claim.
---

# Rigorous Ablation Review

Use this skill to decide whether an ML experiment isolates the claimed causal variable and whether its conclusion matches its evidence.

## Start from the claim

Write the central claim as an estimand before judging implementation. Prefer a contrast such as:

$$
\Delta=E[Y\mid A]-E[Y\mid B].
$$
State exactly what differs between `A` and `B`. If more than one scientifically meaningful variable differs, do not call the comparison a pure ablation.

Distinguish:

- performance claim: one method improves an outcome;
- efficiency claim: it reaches an outcome with fewer tokens, FLOPs, or time;
- stability claim: it reduces prespecified failures or variance;
- mechanism claim: an intermediate process causes the outcome;
- equivalence claim: any difference lies inside a prespecified practical margin.

Each requires different evidence.

## Audit in this order

### 1. Treatment isolation

List every variable changed across arms, including:

- data identities, order, mixture, and boundaries;
- model initialization and architecture;
- batch size, sequence length, tokens/update, and number of updates;
- learning-rate path indexed by steps or tokens;
- optimizer state and reset behavior;
- transition count, block length, pacing, and final training stage;
- evaluation targets, context, decoding, and checkpoint selection;
- wall-clock environment and stopping rule.

Label each difference as intended treatment, necessary consequence, measured mediator, or uncontrolled confound.

### 2. Counterfactual alignment

Ask whether the arms end on the same target task and are evaluated on identical examples and target positions. When studying early training order, prefer a shared final recovery or fine-tuning stage so final-task mismatch is not mistaken for path dependence.

When studying context length, compare predictions for the same target token while changing only available context.

### 3. Data pairing and independence

Verify pairing in code, not only in prose. Check whether the same seed actually produces shared initialization and matched data draws. Do not call samples unique or non-overlapping when sampling with replacement or when windows can overlap.

Pairing improves within-seed comparisons but does not create independent seeds or prove dataset-level generalization.

### 4. Measurement validity

For every metric, identify:

- the target quantity;
- the estimator;
- the data used to fit and evaluate it;
- known bias or leakage;
- whether arms score identical targets;
- whether the metric is sensitive to the proposed effect.

Reject in-sample entropy estimates as evidence of held-out predictability. Reject aggregate metrics as sole evidence when a sparse affected subset could be averaged away.

### 5. Dynamics, endpoint, and cost

Do not infer dynamics from an endpoint alone. When relevant, request:

- learning curves and AUC;
- transition-local measurements;
- time or tokens to a fixed threshold;
- recovery rate after a shared intervention;
- final performance;
- tokens, FLOPs, wall-clock, and memory.

Keep token-matched and compute-matched claims separate when sequence length changes computation.

### 6. Statistics and decision rules

Report raw seed-level values, paired differences, effect sizes, and uncertainty intervals. Treat seed count as a limit on conclusion strength.

Never infer equivalence from `p > 0.05`. An equivalence claim requires a prespecified practical margin and an interval or equivalence test that fits inside it.

For low-compute pilots, allow few seeds to screen for large, directionally consistent effects. Require weaker wording for small or uncertain effects. Prefer a prespecified go/no-go threshold over adding experiments after seeing results.

### 7. Mechanism evidence

An observed correlation or diagnostic is not a mechanism. Require a chain of evidence:

$$
\text{treatment}\rightarrow\text{mediator}\rightarrow\text{outcome}
$$
and, where feasible, an intervention on the proposed mediator. Test the cheapest decisive alternative explanation first. Do not expand into a large factorial experiment until the primary effect survives basic controls.

### 8. Claim calibration

Classify every conclusion as one of:

- confirmed within the tested setting;
- supported but mechanism unresolved;
- exploratory observation;
- no detectable difference under current power;
- practically equivalent within a stated margin;
- contradicted;
- unmeasurable with the current estimator.

Scope claims to the tested model, data, tokenizer, context range, budget, and evaluation unless external-validity experiments justify broader language.

## Produce an actionable review

Return:

1. the estimand and strongest defensible current conclusion;
2. what the design already controls well;
3. confounds ordered by threat to validity;
4. measurement or implementation defects with code evidence when available;
5. the smallest decisive experiment;
6. primary metrics, effect thresholds, and stop/go rules;
7. claims that must be weakened, retracted, or left unknown.

Separate required repairs from optional extensions. Under tight compute, prioritize one high-information intervention, cheap reanalysis of existing logs, and evaluation-only additions before more seeds, larger models, or new datasets.

## Red flags

- The control changes order and transition frequency simultaneously.
- Arms finish on different training tasks but are compared on one arm's final task.
- Same seed is described as paired even though RNG consumption diverges.
- `p > 0.05` is called equivalence.
- A single batch or initialization diagnostic is used to explain final performance.
- A mechanism is named only after the endpoint result is known.
- Hyperparameters or stopping rules are selected using the reported test outcome.
- The report says samples are unique while the code uses replacement.
- Token matching is presented as compute matching without measuring time or FLOPs.
- A polished report is treated as evidence without checking code, raw results, and reproducibility.
