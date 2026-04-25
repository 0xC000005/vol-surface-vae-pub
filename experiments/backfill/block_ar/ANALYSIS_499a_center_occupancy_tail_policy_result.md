# 499a Center-Occupancy + Tail Policy Result

## Hypothesis

498a showed that median-locked residual-width policy preserves conditionality but
cannot fix level KS. 499a therefore added one deployable level-occupancy component:

```text
center map = pre-validation quantile map from 392a sample medians to realized futures
residuals  = 392a residual shapes reattached around mapped centers
tails      = pre-validation asymmetric lower/upper residual scales
```

This was a strong falsifier with `center_alpha=1.0`: if full center occupancy transfer
could not help, weak blends are unlikely to solve the core level-law problem.

## Artifacts

- Evaluator: `experiments/backfill/block_ar/evaluate_499a_center_occupancy_tail_policy.py`
- Result: `results/block_ar/499a_center_occupancy_tail_policy/full11.json`
- Markdown: `results/block_ar/499a_center_occupancy_tail_policy/full11.md`
- Policy: `results/block_ar/499a_center_occupancy_tail_policy/policy.json`

## Result

Score: `5/11`.

Passed:

- surface
- conditionality
- block_ar
- cross_cell_correlation
- pathwise_jump_realism

Failed:

- coverage
- time_series
- cointegration
- regime_coverage
- distributional_fidelity
- mean_reversion

Key metrics:

- coverage90: `0.8192`
- conditional MAE reduction: `7.28%`
- daily-change KS: `23/25`
- level KS: `1/25`
- median-bias cells: `14/25`
- regime layer2: `0/8`
- cointegration worst-cell ratio: `0.105`
- mean-reversion ratio: `0.628`, active pass `41.7%`
- center mean/p95 absolute shift: `0.0155` / `0.0631`
- pathwise max-jump KS: `0.367`

## Mechanism Read

The level-occupancy map failed in the wrong direction. It preserved conditionality
because the center remains history-dependent, but the pre-validation quantile map did
not transfer to validation:

- level KS collapsed from `10/25` to `1/25`;
- median-bias cells fell from `20/25` to `14/25`;
- coverage became undercovered at later horizons, especially cell `(0,0)`;
- mean reversion was weakened enough to fail;
- the mapped center created ceiling/floor and window-floor failures inside the
  distributional suite.

The issue is not that level movement is impossible; it is that unconditional
pre-validation center distribution matching is not stable enough as a deployable policy
on this split.

## Decision

Close naive pre-validation center quantile mapping. The remaining level-occupancy
problem cannot be solved by a broad unconditional center map without damaging structural
law. If the loop continues, the next step should be post-experiment ideation around a
more conservative, state-conditional center policy or a product decision, not another
global marginal quantile map.
