# 496a Center-Preserving Stress Ensemble Result

## Hypothesis

495a showed that a 392a/494a learned stress ensemble preserved structure but failed
conditionality. 496a tested whether that failure was mainly due to stress-member
center drift:

```text
center = median(samples_392a)
residual_shape = ensemble_samples - median(ensemble_samples)
output = center + residual_shape
```

This uses only frozen learned generators and the current history. It does not use
validation futures, calibration tables, or validation-tuned weights.

## Artifacts

- Evaluator: `experiments/backfill/block_ar/evaluate_496a_centered_stress_ensemble.py`
- Result: `results/block_ar/496a_centered_392a_494a_stress_ensemble/full11.json`
- Markdown: `results/block_ar/496a_centered_392a_494a_stress_ensemble/full11.md`

## Result

Score: `6/11`.

Passed:

- surface
- time_series
- block_ar
- cross_cell_correlation
- mean_reversion
- pathwise_jump_realism

Failed:

- coverage
- conditionality
- cointegration
- regime_coverage
- distributional_fidelity

Key metrics:

- coverage90: `0.8909`
- conditional MAE reduction: `3.43%`
- daily KS: `25/25`
- level KS: `13/25`
- median-bias cells: `20/25`
- regime layer2: `0/8`
- cointegration worst-cell ratio: `0.246`
- cross-cell corr ratio: `0.940`
- mean-reversion ratio: `1.017`, active pass `79.2%`
- pathwise max-jump KS: `0.331`

## Mechanism Read

Center preservation did not fix the problem:

- Conditionality worsened from 495a's `4.5%` to `3.43%`, so the issue is not just
  sample-median drift from the stress member.
- Level KS improved slightly to `13/25`, but it still does not reach the `15/25`
  gate.
- Regime layer2 fell back to `0/8`.
- Worst-cell cointegration slipped below the gate.

## Decision

Close center-preserving stress ensembling. It is another below-frontier policy that
preserves many structural diagnostics but cannot learn conditional level/regime
allocation. Do not tune sample ratios or recentering variants.

The next step should now be a true paradigm decision, not another 392a wrapper,
ensemble, or calibration layer.
