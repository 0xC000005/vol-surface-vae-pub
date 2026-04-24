# 406a Interval-Scale Tradeoff

| system | score | failed | cov90 | under70 | over95 | cond MAE | very-small moves | level KS | regime L2 | coint worst |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 392a base | 8/11 | coverage, regime_coverage, distributional_fidelity | 0.868 | 1 | 10 | 5.14% | 0.949 | 10/25 | 0/8 | 0.278 |
| 405a target90 scale | 6/11 | coverage, conditionality, time_series, regime_coverage, distributional_fidelity | 0.922 | 0 | 27 | 3.75% | 0.838 | 11/25 | 1/8 | 0.298 |

## Mechanism Read

Full target-90 interval scaling proves width is an effective actuator: undercoverage disappears, regime layer2 improves from 0/8 to 1/8, cointegration remains valid, and level KS improves slightly. But because the objective targets 90% everywhere, it over-widens many already-safe cells and breaks conditionality plus the small-move profile.

## Decision

Do not tune alpha as the primary next move. The principled calibrated risk objective is a deadband policy: leave cells unchanged if calibration coverage is already inside the evaluator/risk band and use the smallest scale change needed to enter the band. This is narrower than target-90 scaling and should preserve base dynamics better.
