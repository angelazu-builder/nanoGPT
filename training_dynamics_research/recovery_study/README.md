# Context-Length Order Recovery Study

Execution implementation for the preregistered study:
`training_dynamics_research/recovery_study/PREREGISTRATION.md` (Frozen commit: `f1bf7b7`).

---

## Directory Structure

```text
training_dynamics_research/recovery_study/
├── PREREGISTRATION.md   # Frozen experimental specification & decision rules
├── PREREGISTRATION_AMENDMENT_001_FIGURES.md
├── TECHNICAL_REPORT.md  # Corrected results, interpretation, and provenance
├── config.py            # Global hyperparameters, seeds, and decision thresholds
├── schedules.py         # Ascending, descending, and nonmonotonic schedule builders
├── manifests.py         # RNG-decoupled scheduled, recovery, & extended batch manifests
├── evaluation.py        # Zero-redundancy nested evaluation (process panel + anchor panel)
├── runner.py            # Execution runner with rotation, full state ckpts, & extension hook
├── analyze.py           # Multi-horizon statistical analysis, transition shock, AUC, & plots
├── test_study.py        # Unit tests covering manifests, schedules, & Case A-E decision rules
└── README.md            # This documentation
```

Outputs are automatically organized into:

```text
results/recovery_study/
├── run_config.json
├── validation_manifest.npz
├── seed_42/{ascending,descending,nonmonotonic}/
├── seed_43/{ascending,descending,nonmonotonic}/
├── seed_44/{ascending,descending,nonmonotonic}/
├── figures/
│   ├── data/            # Five canonical tidy CSV tables
│   ├── main_01_*.png ... main_05_*.png
│   └── appendix_A1_*.png ... appendix_A5_*.png
└── summary.json
```

---

## Quickstart & Execution Commands

### 1. Run Unit Tests (Schedules, Manifests, & Synthetic Decisions)

```bash
python3 -m unittest training_dynamics_research/recovery_study/test_study.py
```

### 2. Run Smoke Test (10 steps/block, 10 recovery steps, 1 seed)

```bash
python3 -m training_dynamics_research.recovery_study.runner --smoke-test
```

### 3. Run Full Formal Experiment (3 arms × 3 seeds = 9 runs)

```bash
python3 -m training_dynamics_research.recovery_study.runner
```

*CLI Options:*
- `--seed <int>`: Run only a specific seed (e.g. `--seed 42`)
- `--arm <str>`: Run only a specific arm (e.g. `--arm ascending`)
- `--overwrite`: Re-run and overwrite existing completed runs

### 4. Trigger Adaptive Extension (if Case D is met)

If `analyze.py` diagnoses Case D (active recovery, gap still decreasing):

```bash
python3 -m training_dynamics_research.recovery_study.runner --extend
```
This loads each arm's `checkpoint_step_2500.pt`, preserves Adam moments and RNG states, and extends training with clamped `min_lr = 1e-4` up to step 3000 using the preregistered `extended_recovery_manifest`.

### 5. Analyze Results & Generate Plots

```bash
python3 -m training_dynamics_research.recovery_study.analyze
```
Outputs the corrected preregistered summary, five tidy tables, and the complete figure set under `results/recovery_study/figures/`. The interpretation and exact training/analysis/figure provenance are recorded in `TECHNICAL_REPORT.md`.
