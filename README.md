# Angela's nanoGPT

[English](README.md) | [中文](README_CN.md)

A small, inspectable GPT implementation for Apple Silicon, and a three-iteration study of how context-length order affects training dynamics on Tiny Shakespeare.

## Two connected tasks

1. **Build and train nanoGPT.** The root modules implement the model, character/BPE data pipelines, training, evaluation, and text generation.
2. **Study training dynamics.** The model is used as an experimental instrument for controlled context-length curriculum studies.

The second task is the scientific center of the current repository. Start with the [training-dynamics research map](training_dynamics_research/README.md).

## Research journey

| Iteration | Question | What changed | Defensible outcome |
|---|---|---|---|
| 1. Exploratory pilot | Does context-length order matter? | Single-seed schedules plus diagnostic probes | A dramatic descending deficit appeared, but measurement problems made it a clue rather than a conclusion. |
| 2. Paired pilot | Does the deficit survive corrected measurement? | Five paired seeds, shared initialization and data manifests, fixed held-out targets | The descending terminal deficit replicated; curriculum showed no detectable advantage over shuffled or fixed-long. Terminal context remained confounded with order. |
| 3. Recovery study | Is the deficit persistent under a common long-context condition? | Matched block permutations, common `T=256` recovery, multi-horizon process and anchor panels, preregistered decisions | **Large deficit reversed; residual sign reversal unresolved.** |

This history matters. The current conclusion is not that curriculum is universally better, that ordering never matters, or that recency has been uniquely identified.

## Final deliverables

- [Final technical report](training_dynamics_research/recovery_study/TECHNICAL_REPORT_final.md) — full process, audit trail, failures, registered analysis, results, and remaining unknowns.
- [Final paper source](training_dynamics_research/paper_icml2026/paper_final.tex) — named ICML-style preprint.
- [Final paper PDF](output/pdf/context_order_recovery_icml2026_final.pdf).
- [Frozen preregistration](training_dynamics_research/recovery_study/PREREGISTRATION.md).

Only final deliverables use the `_final` suffix. Executable modules and machine-generated artifacts retain stable names so imports and reproduction commands do not break.

## Repository map

```text
.
├── model.py, train.py, dataset*.py, eval.py, chat.py
│   └── core nanoGPT implementation and training interface
├── Day 1/, Day 2/, Day 3/
│   └── early model-building milestones
├── training_dynamics_research/
│   ├── README.md                    # research journey and reading order
│   ├── history/                     # iterations 1–2; historical, not current conclusions
│   ├── recovery_study/              # iteration 3 code, preregistration, final report
│   ├── paper_icml2026/              # final paper source and ICML style files
│   └── research_methodology/        # reusable experiment-review and code-review guidance
├── results/
│   ├── ce3_paired/                  # iteration 2 results
│   └── recovery_study/              # iteration 3 formal results and figures
└── output/pdf/
    └── context_order_recovery_icml2026_final.pdf
```

## Quickstart

```bash
pip install torch tiktoken matplotlib scipy numpy

# Train the base model
python train.py --tokenizer char --block_size 256

# Generate text
python chat.py --temperature 0.8 --top_k 40
```

## Reproduce the final recovery study

```bash
# Scientific-invariant and decision-rule tests
python3 -m unittest training_dynamics_research.recovery_study.test_study

# Formal experiment: 3 arms × 3 paired seeds
python3 -m training_dynamics_research.recovery_study.runner

# Registered analysis, tidy tables, audit figures, and manuscript figures
python3 -m training_dynamics_research.recovery_study.analyze
```

Run these commands from the repository root.

The formal experiment used a 4.8M-parameter character-level Transformer, Tiny Shakespeare, 4,096 target tokens per update, and an Apple M3 with 24 GB unified memory.
