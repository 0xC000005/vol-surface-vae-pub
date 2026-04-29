# 762a Short-Prefix Level-Marginal CRPS

## Hypothesis

761a showed that calibration improves validation only as a recent rolling product layer. 762a tested whether the base learned law can repair median/level allocation directly by adding a generic level-coordinate marginal CRPS term to the active 755a short-prefix AR flow recipe.

The changed axis was intentionally narrow:

- Source checkpoint: `models/backfill/674a_iv_channel_level_alltrain_w005_e3_s6731/best_model.pt`.
- Base recipe: 755a short-prefix generated FM with `free_running_fm_prefix_steps=5`, `free_running_fm_weight=0.2`, and `channel_level_energy_weight=0.05`.
- New term: `level_marginal_crps_weight=0.05`.
- Coordinate: `scaled_delta`, i.e. level path deltas from the last conditioned level divided by the history-only scale.

## Result

| split | score | failed suites | cov90 | calerr | level KS | bias | coint ratio | coint worst | MR | path KS | kurt |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 762a val | 6/11 | coverage, conditionality, regime_coverage, distributional_fidelity, mean_reversion | 0.780 | 0.099 | 13/25 | 14/25 | 0.732 | 0.269 | false | 0.340 | 1.136 |
| 755a val | 6/11 | coverage, conditionality, cointegration, regime_coverage, distributional_fidelity | 0.817 | 0.063 | 15/25 | 15/25 | 0.746 | 0.239 | true | 0.395 | 1.099 |
| 762a train-tail | 6/11 | coverage, conditionality, time_series, cointegration, regime_coverage | 0.824 | 0.054 | 19/25 | 25/25 | 0.458 | 0.198 | true | 0.441 | 1.955 |
| 755a train-tail | 8/11 | conditionality, time_series, regime_coverage | 0.843 | 0.031 | 21/25 | 25/25 | 0.550 | 0.279 | true | 0.490 | 1.943 |

Training objective also regressed versus 755a. The best 762a checkpoint was epoch 1 with validation training objective `2.1569`, versus 755a best `2.0605`.

## Mechanism Read

The level marginal CRPS term is directionally meaningful but not sufficient. On validation, it improves the worst-cell cointegration gate and pathwise KS, but it narrows support and worsens the exact failure it was meant to repair: level-KS falls from `15/25` to `13/25`, median-bias falls from `15/25` to `14/25`, and cov90 falls from `0.817` to `0.780`.

On train-tail, the result is more decisive: 755a was `8/11`, while 762a drops to `6/11`. The new score preserves median-bias cells but damages coverage, cointegration, kurtosis/time-series behavior, and risk-state allocation. This means the bottleneck is not simply absence of a level proper score. The loss creates a central-level versus structural-dynamics trade-off.

## Decision

Reject 762a as a promotion candidate. Keep 755a as the active base learned generator and 760a as a separate calibrated risk-system layer.

Next step should not be another scalar weight on this same loss unless diagnostics justify it. The more principled move is post-experiment analysis or ideation on why base level allocation and structural dynamics trade off under the current readout/objective, before adding another knob.
