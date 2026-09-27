# Preregistration Amendment 001 — Figure and Process-Analysis Contract

## Status and scope

- Study: Context-Length Order Recovery Study
- Base preregistration: `training_dynamics_research/recovery_study/PREREGISTRATION.md`
- Frozen base commit: `f1bf7b7`
- Amendment date: 2026-09-27
- Timing: written before any formal 3-arm × 3-seed result was produced or inspected
- Smoke-test outputs: excluded from all formal analysis and figures

This amendment prespecifies visualization and process-analysis outputs that were discussed during study design but omitted from the frozen preregistration and initial implementation.

It does **not** change:

- research question or hypotheses;
- arms or schedules;
- seeds;
- training budget;
- process or anchor panels;
- evaluation checkpoints;
- estimands;
- decision thresholds;
- stop/go rules;
- extension policy;
- confirmatory versus exploratory status of the nonmonotonic arm.

The purpose is to prevent post-result selection of horizons, checkpoints, plots, color scales, or seed summaries.

## 1. Frozen data structure

### Process panel

- 16 fixed validation sequences
- the first 16 sequences of the anchor manifest
- four evaluation horizons: `32, 64, 128, 256`
- the same final 32 target positions at every horizon
- evaluated at exactly 17 global steps:

```text
250,
500, 501,
750,
1000, 1001,
1250,
1500, 1501,
1750,
2000, 2001, 2010, 2050,
2100, 2250, 2500
```

### Anchor panel

- 32 fixed validation sequences
- the process panel is its first 16 sequences
- four evaluation horizons: `32, 64, 128, 256`
- the same final 32 target positions at every horizon
- evaluated at global steps `2000` and `2500`
- evaluated at step `3000` only if the preregistered one-time extension is triggered

### Required logged values

For every seed, arm, checkpoint, panel, and horizon, save:

```text
seed
arm
global_step
recovery_step
panel
evaluation_horizon
mean_bpc
sequence_bpc[]
current_training_context
```

For anchor checkpoints, save all 32 per-sequence BPC values for every horizon. The first 16 values must exactly reproduce the process-panel statistic from the same forward pass or an explicitly verified equivalent computation.

For context-sensitivity analysis at anchor checkpoints, save per-target losses for horizons 32 and 256 with stable target identifiers. A target identifier must resolve at least:

```text
validation_sequence_index
target_offset_within_scored_window
absolute_validation_position
```

## 2. Tidy analysis tables

Before plotting, export the following machine-readable tables.

### `figures/data/process_horizon_bpc.csv`

One row per:

```text
seed × arm × process checkpoint × horizon
```

Columns:

```text
seed,arm,global_step,recovery_step,current_training_context,
evaluation_horizon,mean_bpc
```

### `figures/data/anchor_horizon_bpc.csv`

One row per:

```text
seed × arm × anchor checkpoint × horizon
```

Columns:

```text
seed,arm,global_step,recovery_step,evaluation_horizon,mean_bpc
```

### `figures/data/anchor_sequence_bpc.csv`

One row per:

```text
seed × arm × anchor checkpoint × horizon × validation sequence
```

### `figures/data/recovery_contrasts.csv`

Contains the prespecified paired contrasts:

\[
\Delta_h^P(k)
=
\mathrm{BPC}_{descending,h}^{P}(2000+k)
-
\mathrm{BPC}_{ascending,h}^{P}(2000+k).
\]

### `figures/data/transition_shocks.csv`

Contains:

\[
S_{a,h,t}
=
\mathrm{BPC}_{a,h}^{P}(t+1)
-
\mathrm{BPC}_{a,h}^{P}(t)
\]

for transitions `500→501`, `1000→1001`, `1500→1501`, and `2000→2001`.

## 3. General plotting policy

All formal plots must follow these rules:

1. Plot all three formal seeds. Raw seed trajectories or points must remain visible whenever space permits.
2. A thick line or larger marker may show the 3-seed mean; it must not replace raw seeds.
3. Do not use per-checkpoint significance stars or search for individually significant timepoints.
4. Do not smooth, interpolate, or fit splines through the 17 measured checkpoints in the main figures.
5. Connect measured checkpoints with straight line segments only for visual continuity.
6. Use identical arm colors in every figure:

   ```text
   ascending    = blue
   descending   = orange
   nonmonotonic = green
   ```

7. Use identical horizon colors wherever horizons are encoded by color:

   ```text
   T=32  = lightest purple
   T=64  = medium-light purple
   T=128 = medium-dark purple
   T=256 = darkest purple
   ```

8. Heatmaps at steps 2000 and 2500 must use the same absolute color limits.
9. Change heatmaps must use a diverging color scale centered exactly at zero with symmetric limits determined from the maximum absolute displayed change.
10. Axis limits must not be selected separately to exaggerate one arm or horizon. Facets representing the same metric must share y-axis limits.
11. Missing or failed formal runs must be visibly marked. Do not silently average over a smaller seed set.
12. Figures may not be omitted because they look uninteresting or contradict the narrative.
13. Any additional figure designed after formal results are inspected must be labeled `Post-hoc exploratory` in both filename metadata and report caption.

## 4. Main Figure 1 — Training trajectories by evaluation horizon

Output:

```text
figures/main_01_training_trajectories_by_horizon.png
```

Layout: `2 × 2` facets.

```text
Panel A: evaluation horizon T=32
Panel B: evaluation horizon T=64
Panel C: evaluation horizon T=128
Panel D: evaluation horizon T=256
```

For each panel:

- x-axis: global training step;
- y-axis: process-panel validation BPC;
- thin semi-transparent lines: individual seeds;
- thick lines: 3-seed arm means;
- line color: training arm;
- vertical reference lines: block boundaries at steps 500, 1000, 1500, and 2000;
- no smoothing.

Purpose: determine when arm differences emerge and whether they are shared across evaluation horizons.

Inferential status: prespecified secondary process analysis.

## 5. Main Figure 2 — Recovery gap by evaluation horizon

Output:

```text
figures/main_02_recovery_gap_by_horizon.png
```

For recovery checkpoints, compute:

\[
\Delta_h^P(k)
=
\mathrm{BPC}_{descending,h}^{P}(2000+k)
-
\mathrm{BPC}_{ascending,h}^{P}(2000+k).
\]

Plot:

- x-axis: recovery step `k`;
- y-axis: descending minus ascending BPC;
- one horizon color for each of `32, 64, 128, 256`;
- thin semi-transparent seed-level lines or connected points;
- thick 3-seed mean lines;
- horizontal reference line at zero;
- vertical reference markers at `k=0`, `k=250`, and `k=500`.

Purpose: distinguish horizon-specific recency, broad degradation, slow recovery, and persistent residuals.

Inferential status: the `T=256` anchor endpoints remain confirmatory; the four process trajectories are prespecified secondary evidence.

## 6. Main Figure 3 — Anchor heatmap before recovery

Output:

```text
figures/main_03_anchor_heatmap_step2000.png
```

Matrix:

```text
rows    = ascending, descending, nonmonotonic
columns = T=32, T=64, T=128, T=256
values  = 3-seed mean anchor BPC at step 2000
```

Requirements:

- display the numerical BPC value in every cell;
- use the same absolute color scale as Main Figure 4;
- preserve the registered row and column order;
- caption states that nonmonotonic is exploratory.

Purpose: visualize the context-performance profile immediately before common recovery.

## 7. Main Figure 4 — Anchor heatmap after recovery

Output:

```text
figures/main_04_anchor_heatmap_step2500.png
```

Use the same matrix definition, row order, column order, numerical annotations, and absolute color limits as Main Figure 3.

Purpose: show whether the three context-performance profiles converge after 500 matched `T=256` recovery steps.

## 8. Main Figure 5 — Pre-to-post anchor change

Output:

```text
figures/main_05_pre_post_change_heatmap.png
```

For arm `a` and horizon `h`, compute:

\[
\Delta\mathrm{BPC}_{a,h}
=
\mathrm{BPC}_{a,h}^{A}(2500)
-
\mathrm{BPC}_{a,h}^{A}(2000).
\]

Matrix:

```text
rows    = ascending, descending, nonmonotonic
columns = T=32, T=64, T=128, T=256
values  = 3-seed mean pre-to-post BPC change
```

Requirements:

- diverging scale centered at zero;
- symmetric color limits;
- numerical change printed in every cell;
- negative values labeled as improvement and positive values as degradation.

Purpose: identify which arm and evaluation horizon changes most during matched recovery.

## 9. Appendix Figure A1 — Transition-local shock

Output:

```text
figures/appendix_A1_transition_shock.png
```

Compute process-panel changes at:

```text
500→501
1000→1001
1500→1501
2000→2001
```

Show all arms and horizons using grouped bars or fixed-layout small multiples. Raw seed values must be visible.

Purpose: describe immediate validation response to a context transition. This figure does not by itself establish an optimizer mechanism.

Inferential status: prespecified secondary process analysis.

## 10. Appendix Figure A2 — Context profiles

Output:

```text
figures/appendix_A2_context_profiles.png
```

Two panels:

```text
step 2000
step 2500
```

For each panel:

- x-axis: evaluation horizon `32, 64, 128, 256` on an ordered categorical or log2 scale;
- y-axis: anchor BPC;
- lines: arms;
- raw seed points visible;
- shared y-axis limits.

Purpose: provide a line-based view of the same anchor profiles shown in Main Figures 3 and 4.

## 11. Appendix Figure A3 — Paired seed endpoints

Output:

```text
figures/appendix_A3_paired_seed_endpoints.png
```

For each seed, show paired values of:

\[
\Delta^A(0)
\quad\text{and}\quad
\Delta^A(500).
\]

Connect each seed's two values with a line. Overlay the mean only as a secondary marker.

Purpose: expose direction consistency and prevent the 3-seed mean from hiding heterogeneous responses.

Inferential status: confirmatory endpoint visualization.

## 12. Appendix Figure A4 — Context-alignment contrast

Output:

```text
figures/appendix_A4_context_alignment.png
```

Define:

\[
A^P(k)
=
\Delta_{256}^P(k)
-
\Delta_{32}^P(k).
\]

Plot individual seeds and the 3-seed mean over recovery checkpoints.

Purpose: show whether the descending-versus-ascending difference becomes less dependent on evaluation horizon during common recovery.

Inferential status: prespecified secondary analysis.

## 13. Appendix Figure A5 — Nonmonotonic exploratory contrasts

Output:

```text
figures/appendix_A5_nonmonotonic_exploratory.png
```

Compute:

\[
\Delta_{N-A,h}^{P}(k)
=
\mathrm{BPC}_{nonmonotonic,h}^{P}(k)
-
\mathrm{BPC}_{ascending,h}^{P}(k)
\]

and:

\[
\Delta_{D-N,h}^{P}(k)
=
\mathrm{BPC}_{descending,h}^{P}(k)
-
\mathrm{BPC}_{nonmonotonic,h}^{P}(k).
\]

The title and caption must contain:

```text
Exploratory — one prespecified nonmonotonic permutation
```

Do not generalize from this figure to nonmonotonic schedules as a class.

## 14. Figures after an adaptive extension

If the preregistered extension is triggered:

- Main Figure 2 extends through recovery step `k=1000`;
- Main Figures 3 and 4 remain step 2000 and 2500 anchor snapshots;
- add, rather than substitute, an extension anchor heatmap:

  ```text
  figures/appendix_A6_extension_anchor_step3000.png
  ```

- Appendix Figure A3 adds \(\Delta^A(1000)\) as a third paired point;
- Appendix Figure A4 extends through `k=1000`;
- do not remove the step-2500 results after observing the extension.

## 15. Figure-generation implementation requirements

The analysis implementation must:

1. validate that all nine formal runs are present before generating formal aggregate figures;
2. validate that all 17 process checkpoints and all four horizons exist for every run;
3. validate anchor steps 2000 and 2500 for every run;
4. normalize JSON horizon keys to integers before sorting or plotting;
5. fail loudly on missing seeds, arms, checkpoints, horizons, or non-finite values;
6. write plot-data CSV files before writing images;
7. create deterministic figures from saved data without rerunning models;
8. use a non-interactive plotting backend;
9. save at least PNG outputs with sufficient resolution for the technical report;
10. close figures after saving to avoid memory accumulation;
11. record the code commit used to generate the figures;
12. never include smoke-test results in formal tables or figures.

Recommended entry point:

```bash
python -m training_dynamics_research.recovery_study.analyze --make-figures
```

## 16. Acceptance checks

Before formal training, tests should verify:

- process-panel sequence BPC is saved for every horizon;
- anchor-panel sequence BPC is saved for every horizon;
- the first 16 anchor sequence values reproduce the process statistic at anchor checkpoints;
- all 17 checkpoints produce a `3 arms × 4 horizons` matrix;
- plot-data table row counts match the frozen design;
- step-2000 and step-2500 heatmaps share identical limits;
- change heatmap limits are symmetric around zero;
- the plotting command fails when a formal seed or checkpoint is missing;
- repeated plotting from the same result files produces identical plot-data tables.

## 17. Reporting boundary

The existence of multiple horizons and checkpoints does not create multiple new confirmatory hypotheses. The primary endpoint and stop/go rules remain those in the base preregistration.

The figures are intended to distinguish process patterns and expose contradictions, not to guarantee a significant finding. Any conclusion must still be calibrated as:

- confirmed within the tested setting;
- supported but mechanism unresolved;
- exploratory;
- no detectable difference under current resolution;
- contradicted;
- or unresolved.
