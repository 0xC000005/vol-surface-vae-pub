# 761a Calibration Transfer Assessment

## Context

760a produced the best current validation calibrated-system score (`7/11`) but failed the no-leakage train-tail diagnostic (`5/11`) when calibrated on older train windows. This separates two questions:

- Is rolling calibration useful for the risk-manager product?
- Is calibration solving the publishable learned conditional law?

## Evidence

| run | calibration | eval | score | cov90 | calerr | level KS | bias | coint worst | MR | path KS | kurt |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 755a base | none | val | 6/11 | 0.817 | 0.063 | 15/25 | 15/25 | 0.239 | true | 0.395 | 1.099 |
| 760a rolling calibrated | train_tail | val | 7/11 | 0.874 | 0.014 | 15/25 | 15/25 | 0.328 | true | 0.489 | 0.968 |
| 760a transfer diagnostic | train | train_tail | 5/11 | 0.872 | 0.007 | 23/25 | 25/25 | 0.342 | true | 0.560 | 1.747 |

## Assessment

Rolling calibration is product-useful but scientifically limited.

- It is useful because recent calibration improves validation coverage and cointegration without breaking mean reversion or time-series realism.
- It is limited because calibration fitted on older train windows does not transfer cleanly to train-tail; pathwise and kurtosis realism degrade.
- It does not fix the base model's median-bias failure on validation.
- It should be reported as a calibrated risk system, not as the base learned conditional law.

## Decision

Do not keep iterating calibration shape as the main research line. The best clean base model remains 755a, and the best current calibrated-system validation result is 760a.

The next base-model research should target median/level allocation directly inside the learned law. A principled direction is a level-coordinate proper scoring term that targets central location and support jointly, without post-hoc shifting. This should be framed as improving the learned conditional path law; calibration remains a separately reported product layer.
