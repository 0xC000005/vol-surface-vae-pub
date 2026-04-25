# 495a 392a/494a Stress Ensemble Result

## Hypothesis

Test one no-validation-tuning learned-law ensemble after 494a:

- `392a` supplies the best conditional geometry and remains the `8/11` frontier.
- `494a` supplies interval-score stress samples with higher aggregate coverage.

The ensemble uses `32` samples from 392a and `16` samples from 494a. This is deployable
model averaging over frozen learned generators, not a calibration table and not a
validation-future oracle.

## Artifacts

- Evaluator: `experiments/backfill/block_ar/evaluate_442a_learned_checkpoint_ensemble.py`
- Result: `results/block_ar/495a_392a_494a_stress_ensemble_32_16/full11.json`
- Markdown: `results/block_ar/495a_392a_494a_stress_ensemble_32_16/full11.md`

## Result

Score: `7/11`.

Passed:

- surface
- time_series
- block_ar
- cointegration
- cross_cell_correlation
- mean_reversion
- pathwise_jump_realism

Failed:

- coverage
- conditionality
- regime_coverage
- distributional_fidelity

Key metrics:

- coverage90: `0.888`
- calibration error: `0.004`
- conditional MAE reduction: `4.5%`
- daily KS: `25/25`
- level KS: `12/25`
- median-bias cells: `20/25`
- bias magnitude: `25/25`
- regime layer2: `1/8`
- cointegration worst-cell ratio: `0.278`
- corr ratio: `0.956`
- MR ratio: `1.001`
- pathwise max-jump KS: `0.382`

## Mechanism Read

The ensemble produces a useful intermediate distribution but does not beat the
frontier:

- It improves aggregate calibration error and eliminates most severe undercoverage,
  but late-horizon overcoverage still fails the 95% cap.
- It preserves the important structural suites: cointegration, correlation,
  mean-reversion, and pathwise realism.
- It dilutes conditionality below the gate, matching the earlier 442a ensemble
  and density-ratio patterns.
- It does not move level KS past `12/25`, far short of the `15/25` gate.

## Decision

Close learned-law averaging as a primary route. The result is useful evidence that
the 494a stress member is not structurally toxic, but model averaging still cannot
learn the missing conditional level/regime allocation.

The next iteration should be a paradigm decision rather than another ensemble
weight or local objective. The evidence now points to either a genuinely new core
with native joint level-allocation dynamics, or an explicit admission that
`11/11` requires policy calibration beyond a pure learned conditional law.
