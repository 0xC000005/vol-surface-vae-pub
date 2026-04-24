# 408a Calibration Cap Analysis

| run | score | failed | cov90 | cov cells fail | cond MAE | small moves | level KS | regime L2 | coint worst | scale min/med/max |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 392a_base | 8/11 | coverage, regime_coverage, distributional_fidelity | 0.868 | 11 | 5.14% | 0.949 | 10/25 | 0/8 | 0.278 | n/a |
| 405a_target90 | 6/11 | coverage, conditionality, time_series, regime_coverage, distributional_fidelity | 0.922 | 27 | 3.75% | 0.838 | 11/25 | 1/8 | 0.298 | 0.65/1.20/1.45 |
| 407a_deadband | 7/11 | coverage, conditionality, regime_coverage, distributional_fidelity | 0.870 | 6 | 4.20% | 0.949 | 12/25 | 1/8 | 0.263 | 0.80/1.00/1.30 |

## Mechanism Read

The interval calibration branch is directionally meaningful but capped as a primary route. Deadband scaling improves the coverage edge count, level KS, and regime layer2 relative to the base while preserving time-series and cointegration, but the same width-only actuator lowers conditional MAE below the gate and still leaves six coverage violations, seven regime layer2 failures, and a three-cell level-KS deficit.

## Decision

Close interval calibration as the primary autoresearch path. It can remain a reportable policy-calibration ablation, but continuing it would require more cell/regime-specific knobs. The next principled route should return to the learned base law and attack long-horizon level occupancy/regime allocation during training, not by post-hoc width manipulation.
