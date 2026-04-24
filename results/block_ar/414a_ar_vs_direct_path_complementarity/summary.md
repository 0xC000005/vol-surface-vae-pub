# 414a AR vs Direct Path Complementarity

| run | score | failed | cov under/over | cond | coint worst | level KS | corr ratio | MR | path KS |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|
| 392a_ar_frontier | 8/11 | coverage, regime_coverage, distributional_fidelity | 1/10 | 5.14% | 0.278 | 10/25 | 0.963 | True | 0.373 |
| 413a_direct_path | 4/11 | coverage, conditionality, time_series, cointegration, regime_coverage, cross_cell_correlation, mean_reversion | 7/1 | 3.98% | 0.132 | 20/25 | 0.486 | False | 0.290 |

Suite-union pass count: `9/11`.

## Mechanism Read

392a and 413a expose a genuine factorization split. The AR model owns local/structural dynamics: conditionality, time-series, cointegration, cross-cell correlation, mean reversion, and pathwise realism. The direct path model owns level occupancy: daily KS, level KS, median fraction, and bias magnitude all pass strongly. Both still fail coverage and regime layer2, but their coverage errors have opposite geometry: 392a has mostly over-95 cells, while 413a has mostly under-70 cells.

## Decision

Do one bounded diagnostic mixture, not as the final architecture but as a mechanism test. A fixed mostly-AR sample mixture can test whether the two learned laws contain complementary support that could later be distilled into one clean model. If the mixture cannot improve beyond 8/11 or damages structural passes, close mixture/ensemble work immediately.
