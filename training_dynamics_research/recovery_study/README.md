# Context-Length Order Recovery Study

Execution implementation for the preregistered study:
`training_dynamics_research/recovery_study/PREREGISTRATION.md` (Frozen commit: `f1bf7b7`).

---

## Directory Structure

```text
training_dynamics_research/recovery_study/
├── PREREGISTRATION.md   # Frozen experimental specification & decision rules
├── config.py            # Global hyperparameters, seeds, and decision thresholds
├── schedules.py         # Ascending, descending, and nonmonotonic schedule builders
├── manifests.py         # RNG-decoupled scheduled & recovery batch manifests
├── evaluation.py        # Nested validation (process panel + anchor panel) & sensitivity probe
├── runner.py            # Execution runner with seed rotation order & MPS sync
├── analyze.py           # Statistical analysis, paired differences, and stop/go decision rules
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
└── summary.json
```

---

## Quickstart & Execution Commands

### 1. Run Unit Tests (Schedules, Manifests, & Verification)

```bash
python -m unittest training_dynamics_research/recovery_study/test_study.py
```

### 2. Run Smoke Test (10 steps/block, 10 recovery steps, 1 seed)

```bash
python -m training_dynamics_research.recovery_study.runner --smoke-test
```

### 3. Run Full Formal Experiment (3 arms × 3 seeds = 9 runs)

```bash
python -m training_dynamics_research.recovery_study.runner
```

*Note: Execution order rotates across seeds to prevent order-of-execution bias:*
- `Seed 42`: ascending → descending → nonmonotonic
- `Seed 43`: descending → nonmonotonic → ascending
- `Seed 44`: nonmonotonic → ascending → descending

### 4. Analyze Results & Evaluate Preregistered Stop/Go Rules

```bash
python -m training_dynamics_research.recovery_study.analyze
```
