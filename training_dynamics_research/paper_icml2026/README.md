# ICML 2026 manuscript

This directory contains the paper manuscript corresponding to the preregistered context-length order recovery study. It is intentionally separate from the audit-oriented technical report.

The manuscript is currently configured as a named preprint for Anqi Zu, University of Oxford (`angela.zoo@foxmail.com`). Remove the `preprint` option from `icml2026` before producing an anonymous review submission.

## Build

From this directory:

```bash
tectonic paper_final.tex
```

The manuscript uses the official ICML 2026 style and the versioned `v3_r3` vector figures. The registered audit artifacts remain under `results/recovery_study/`.

The compiled manuscript is `paper_final.pdf`; the repository-level deliverable is `output/pdf/context_order_recovery_icml2026_final.pdf`.

## Scientific status

The primary result is **large deficit reversed; residual sign reversal unresolved**. The paper does not claim equivalence, identify a unique recovery mechanism, or generalize the exploratory nonmonotonic schedule.
