# 416a Next Direct-Path Repair

| run | score | failed | cond | coint worst | level KS | corr ratio | MR | path KS |
|---|---:|---|---:|---:|---:|---:|---|---:|
| 339a_axial_old | 4/11 | coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism | 2.55% | 0.250 | 8/25 | 0.649 | False | 0.522 |
| 339b_transformer_old | 4/11 | coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism | 4.40% | 0.289 | 5/25 | 0.776 | False | 0.499 |
| 413a_axial_recent | 4/11 | coverage, conditionality, time_series, cointegration, regime_coverage, cross_cell_correlation, mean_reversion | 3.98% | 0.132 | 20/25 | 0.486 | False | 0.290 |
| 415a_mix | 7/11 | coverage, cointegration, regime_coverage, distributional_fidelity | 5.37% | 0.175 | 12/25 | 0.962 | True | 0.374 |

## Mechanism Read

The mixture diagnostic should be closed, but 413a should not be discarded as mere failure. Recent quantile framing plus direct path FM solved level occupancy. The missing piece is structural coupling. The old 339b Transformer path mixer had better cointegration and cross-cell structure than old 339a, but it never received the recent-score framing that made 413a's level KS jump to 20/25.

## Decision

Run one final direct-path repair: recent-quantile adaptation of the 339b Transformer path-flow checkpoint. This is not a depth/head sweep; it is the single missing cross of two known mechanisms: 339b's stronger joint mixer and 413a's recent score framing. If it fails to beat the frontier or at least preserve distributional fidelity while improving structural suites, close direct path flow.
