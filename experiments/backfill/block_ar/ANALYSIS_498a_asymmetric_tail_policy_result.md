# 498a Median-Locked Asymmetric Tail Policy Result

## Hypothesis

After 497a shifted to separated risk-policy reporting, 498a tested the cleanest
deployable interval policy:

```text
base law = frozen 392a samples
center   = frozen 392a sample median
policy   = pre-validation lower/upper residual scales by regime/horizon/cell
output   = center + asymmetric scaled residuals, median-locked back to center
```

The goal was to correct directional under/overcoverage while preserving conditionality.

## Artifacts

- Evaluator: `experiments/backfill/block_ar/evaluate_498a_asymmetric_tail_policy.py`
- Result: `results/block_ar/498a_asymmetric_tail_policy/full11.json`
- Markdown: `results/block_ar/498a_asymmetric_tail_policy/full11.md`
- Policy: `results/block_ar/498a_asymmetric_tail_policy/policy.json`

## Result

Score: `6/11`.

Passed:

- surface
- conditionality
- block_ar
- cross_cell_correlation
- mean_reversion
- pathwise_jump_realism

Failed:

- coverage
- time_series
- cointegration
- regime_coverage
- distributional_fidelity

Key metrics:

- coverage90: `0.8923`
- conditional MAE reduction: `5.04%`
- regime layer2: `0/8`
- daily-change KS: `25/25`
- level KS: `10/25`
- median-bias cells: `20/25`, bias magnitude `25/25`
- cointegration worst-cell ratio: `0.246`
- cross-cell corr ratio: `0.952`
- mean-reversion ratio: `1.058`, active pass `79.2%`
- pathwise max-jump KS: `0.176`

## Mechanism Read

The falsifier was informative:

- Median locking protected conditionality; this is the first recent policy layer that
  preserved the `>5%` MAE-reduction gate.
- But median locking also preserved the level-law defect: level KS stayed `10/25`,
  exactly the 392a frontier value.
- Asymmetric tail scaling improved aggregate interval calibration, but per-cell
  overcoverage remained at h1, h14, and h30, and regime layer2 remained `0/8`.
- The tail policy made the generated daily-change distribution too active at the
  smallest move threshold: `|dIV|<=0.005` ratio `0.866`, causing the time-series suite
  to fail.
- Cointegration was a near miss (`0.246` versus `0.25` worst-cell gate), which is not
  the primary failure mechanism.

## Decision

Close interval-only policy calibration as insufficient for `11/11`. It can preserve
conditionality, but it cannot fix distributional fidelity because the binding level KS
failure is a median/level-occupancy problem.

The next step should target level occupancy explicitly while trying to retain the 498a
lesson: protect conditionality by keeping any level movement low-dimensional,
history-only, and separately reported as policy calibration. A pure residual-width
policy is capped.
