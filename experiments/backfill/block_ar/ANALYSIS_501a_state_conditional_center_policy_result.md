# 501a State-Conditional Center Policy Result

## Hypothesis

500a proposed the most conservative remaining level-occupancy policy:

```text
base center     = 392a sample median
target residual = realized future - base center
features        = low-dimensional history summaries
center policy   = ridge residual prediction
shrinkage       = selected only by pre-validation holdout MAE
tails           = asymmetric residual scales around shifted center
```

This tests whether the missing level law is predictably state-conditional rather than
an unconditional marginal calibration problem.

## Artifacts

- Evaluator: `experiments/backfill/block_ar/evaluate_501a_state_conditional_center_policy.py`
- Result: `results/block_ar/501a_state_conditional_center_policy/full11.json`
- Markdown: `results/block_ar/501a_state_conditional_center_policy/full11.md`
- Policy: `results/block_ar/501a_state_conditional_center_policy/policy.json`

## Result

Score: `5/11`.

Passed:

- surface
- block_ar
- cross_cell_correlation
- mean_reversion
- pathwise_jump_realism

Failed:

- coverage
- conditionality
- time_series
- cointegration
- regime_coverage
- distributional_fidelity

Key metrics:

- selected center alpha: `0.0`
- holdout base/best MAE: `0.03107` / `0.03107`
- validation center shift: `0.0`
- coverage90: `0.8933`
- conditional MAE reduction: `3.97%`
- daily-change KS: `25/25`
- level KS: `11/25`
- median-bias cells: `20/25`
- regime layer2: `0/8`
- cointegration worst-cell ratio: `0.222`
- mean-reversion ratio: `1.056`, active pass `79.2%`
- pathwise max-jump KS: `0.191`

## Mechanism Read

The chronological pre-validation holdout rejected the ridge center residual model:
`alpha=0.0` was the best MAE choice. All nonzero shrinkage values made holdout center
MAE worse. That is the important result.

The final system therefore reduced to an asymmetric-tail policy under a new random
sample draw. It did not improve the frontier and even missed conditionality on this
draw. Level KS was only `11/25`, still far from the `15/25` gate.

## Decision

Close state-conditional linear center policy as a deployable level-occupancy fix. The
available pre-validation history features do not predict center residuals well enough
to justify shifting the 392a median.

The clean conclusion is now stronger: all deployable policy-calibration attempts around
392a are below the `8/11` learned-law frontier, and the only known `11/11` result remains
oracle-feasible but not deployable. The next HEAD step should be a paradigm/product
decision, not another center or residual calibration variant.
