# 617a 616a Student-t temperature diagnostic

## Context

616a Student-t likelihood improved several Gaussian-NLL failures but remained too wide and uneven. 617a tested whether moderate sampler down-temperature could preserve the Student-t likelihood gains while fixing surface, overcoverage, and tail-scale imbalance.

## Runs

Checkpoint:

- `models/backfill/616a_joint38_ar_studentt_nll_e8_w2048_s616/best_model.pt`

Diagnostics:

- temp `0.75`: `results/autoresearch/617a_616a_studentt_temperature_diagnostic/temp075_full11.json`
- temp `0.85`: `results/autoresearch/617a_616a_studentt_temperature_diagnostic/temp085_full11.json`
- temp `0.90`: `results/autoresearch/617a_616a_studentt_temperature_diagnostic/temp090_full11.json`

Baseline temp `1.00` is the 616a full-suite result.

## Results

| run | score | surface | cov90 | cond MAE | cat. undercov | daily KS | level KS | bias cells | kurtosis | coin worst | MR ratio | path KS | q99 cells |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 616a temp 1.00 | 4/11 | fail | 92.2% | 7.9% | 3.6% | 14/25 | 0/25 | 21/25 | 0.452 | 0.545 | 0.427 | 0.403 | 6/25 |
| 617a temp 0.75 | 5/11 | pass | 90.4% | 8.4% | 4.0% | 19/25 | 1/25 | 20/25 | 0.539 | 0.388 | 0.324 | 0.551 | 9/25 |
| 617a temp 0.85 | 5/11 | pass | 91.3% | 5.9% | 3.8% | 17/25 | 0/25 | 21/25 | 0.487 | 0.507 | 0.369 | 0.500 | 8/25 |
| 617a temp 0.90 | 5/11 | pass | 91.6% | 9.2% | 3.7% | 17/25 | 0/25 | 21/25 | 0.474 | 0.343 | 0.370 | 0.479 | 6/25 |

## Mechanism Read

Student-t temperature improves the best likelihood-path score from `4/11` to `5/11`, but does not solve the core pathology.

What improves:

- surface validity recovers for all tested down-temperatures;
- conditionality remains passable;
- persistent severe undercoverage remains under the 5% gate;
- daily-change KS passes at all tested down-temperatures;
- median-bias cells pass at all tested down-temperatures;
- cointegration and cross-cell correlation remain healthy;
- pathwise aggregate max-jump KS passes at `0.90`.

What remains broken:

- coverage suite still fails due per-cell overcoverage upper caps and isolated lower-tail cells;
- regime layer2 remains `0/8`;
- level KS remains effectively dead (`0/25` or `1/25`);
- mean reversion remains weak (`0.324` to `0.370`);
- per-cell q99 jump scale remains extremely imbalanced (`6/25` to `9/25`);
- time-series suite still fails because aggregate kurtosis remains too low.

Temperature therefore cannot fix the likelihood model. The issue is conditional placement and per-cell scale geometry, not only global innovation amplitude.

## Decision

Close Student-t sampler-temperature as a deployable fix.

The next move should target conditional mean/placement directly while staying in the likelihood-trained one-model paradigm. A clean option is to add an explicit one-step residual mean objective alongside NLL, using the same predicted transition mean and no separate IV/factor path. This is not a post-hoc correction; it asks the likelihood model to improve conditional center placement so level occupancy and mean reversion are not left entirely to broad stochastic coverage.
