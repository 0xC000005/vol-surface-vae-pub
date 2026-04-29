# 763a Level-CRPS Trade-Off Attribution

## Decision

The level marginal CRPS term is a proper score but its current scale narrows generated support and competes with structural dynamics. It improves some validation cointegration/pathwise geometry while worsening coverage, level-KS, median-bias, mean reversion, and train-tail robustness.

## Score Comparison

| split | model | score | failed | cov90 | level KS | bias | coint worst | MR | kurt |
|---|---|---:|---|---:|---:|---:|---:|---|---:|
| val | 755a | 6/11 | coverage, conditionality, cointegration, regime_coverage, distributional_fidelity | 0.817 | 15/25 | 15/25 | 0.239 | true | 1.099 |
| val | 762a | 6/11 | coverage, conditionality, regime_coverage, distributional_fidelity, mean_reversion | 0.780 | 13/25 | 14/25 | 0.269 | false | 1.136 |
| train-tail | 755a | 8/11 | conditionality, time_series, regime_coverage | 0.843 | 21/25 | 25/25 | 0.279 | true | 1.943 |
| train-tail | 762a | 6/11 | coverage, conditionality, time_series, cointegration, regime_coverage | 0.824 | 19/25 | 25/25 | 0.198 | true | 1.955 |

## Objective Scale

- 755a best val objective: `2.0605` at epoch `1`.
- 762a best val objective: `2.1569` at epoch `1`.
- 762a weighted level-CRPS contribution is `0.0985`.
- 762a weighted channel-level-energy contribution is `0.0138`.
- Level-CRPS contribution is `7.2x` channel-level-energy contribution.

## Spread Effect

- Validation sample normalized std changed `0.844 -> 0.807` (-4.4%).
- Validation sample level std changed `0.729 -> 0.686` (-5.9%).

## Largest Validation Grid Deltas

- `coverage_h30`: mean delta `-0.0547`, median delta `-0.0385`, max abs delta `-0.1497` at cell `[1, 2]`.
- `level_ks`: mean delta `+0.0138`, median delta `+0.0184`, max abs delta `+0.0501` at cell `[2, 3]`.
- `median_above_frac`: mean delta `+0.0465`, median delta `+0.0531`, max abs delta `+0.0803` at cell `[3, 1]`.
- `cointegration_ratio`: mean delta `+0.0107`, median delta `-0.0278`, max abs delta `+0.7368` at cell `[3, 1]`.

## Next Step

Do not stack another scalar level loss. Analyze objective/readout interaction or consider a cleaner reformulation that avoids adding proper scores with uncontrolled relative scale.
