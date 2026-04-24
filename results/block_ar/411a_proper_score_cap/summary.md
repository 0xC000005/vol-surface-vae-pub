# 411a Proper-Score Fine-Tune Cap

| run | score | failed | cov90 | under/over | cond MAE | coint worst | level KS | regime L2 | path KS |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 385a_recent_fm | 8/11 | coverage, regime_coverage, distributional_fidelity | 0.879 | 2/12 | 6.08% | 0.333 | 5/25 | 0/8 | 0.370 |
| 392a_energy_w005 | 8/11 | coverage, regime_coverage, distributional_fidelity | 0.868 | 1/10 | 5.14% | 0.278 | 10/25 | 0/8 | 0.373 |
| 393a_energy_w01 | 7/11 | coverage, conditionality, regime_coverage, distributional_fidelity | 0.847 | 0/2 | 3.96% | 0.250 | 12/25 | 0/8 | 0.414 |
| 410a_marginal_crps_w005 | 7/11 | coverage, cointegration, regime_coverage, distributional_fidelity | 0.874 | 0/15 | 5.60% | 0.219 | 10/25 | 1/8 | 0.375 |

## Mechanism Read

Simple free-running proper-score fine-tuning is capped as a primary route. Weak multivariate energy gave the best score by trading part of 385a's conditionality margin for better level occupancy. Stronger energy improved level KS further but lost conditionality/coverage. Marginal CRPS preserved conditionality and improved calibration error, but did not move level KS and lost worst-cell cointegration. The common pattern is objective geometry, not architecture collapse: scalar fine-tune losses can move one failed suite but do not produce the joint conditional level/regime law needed for 11/11.

## Decision

Keep 392a as the active 8/11 frontier and close simple proper-score fine-tune losses as the primary route. The next iteration should be a paradigm shift in the base likelihood/representation, not another energy/CRPS weight or post-hoc calibration branch.
