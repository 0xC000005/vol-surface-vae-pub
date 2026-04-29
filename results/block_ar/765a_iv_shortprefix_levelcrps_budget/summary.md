# 765a Budget-Matched Level CRPS Replacement

## Hypothesis

764a selected the cleanest falsifier after 762a/763a: keep the 755a short-prefix generated-prefix FM recipe, but replace the raw channel-level energy term with a budget-matched proper level-score term instead of stacking another large objective component.

Recipe:

- Source checkpoint: `models/backfill/674a_iv_channel_level_alltrain_w005_e3_s6731/best_model.pt`.
- Short-prefix generated FM: `free_running_fm_weight=0.2`, `free_running_fm_prefix_steps=5`.
- Removed raw level energy: `channel_level_energy_weight=0.0`.
- Added proper level score: `level_marginal_crps_weight=0.007`.
- Coordinate: `level_marginal_crps_coordinate=scaled_delta`.

The weight came from 763a objective-budget attribution: `0.05 * 0.275 / 1.970 = 0.00698`.

## Result

| split | score | failed suites | cov90 | calerr | level KS | bias mag | coint ratio | coint worst | MR full active | path KS | kurt |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 765a val | 6/11 | coverage, conditionality, regime_coverage, distributional_fidelity, mean_reversion | 0.814 | 0.068 | 14/25 | 22/25 | 0.743 | 0.313 | 0.577 | 0.383 | 1.106 |
| 755a val | 6/11 | coverage, conditionality, cointegration, regime_coverage, distributional_fidelity | 0.817 | 0.063 | 15/25 | 22/25 | 0.746 | 0.239 | 0.738 | 0.395 | 1.099 |
| 762a val | 6/11 | coverage, conditionality, regime_coverage, distributional_fidelity, mean_reversion | 0.780 | 0.099 | 13/25 | 22/25 | 0.732 | 0.269 | 0.618 | 0.340 | 1.136 |
| 765a train-tail | 7/11 | conditionality, time_series, regime_coverage, mean_reversion | 0.847 | 0.031 | 22/25 | 22/25 | 0.518 | 0.279 | 0.614 | 0.464 | 1.936 |
| 755a train-tail | 8/11 | conditionality, time_series, regime_coverage | 0.843 | 0.031 | 21/25 | 22/25 | 0.550 | 0.279 | 0.812 | 0.490 | 1.943 |
| 762a train-tail | 6/11 | coverage, conditionality, time_series, cointegration, regime_coverage | 0.824 | 0.054 | 19/25 | 23/25 | 0.458 | 0.198 | 0.668 | 0.441 | 1.955 |

Training selected epoch 1 with best validation training objective `2.0383`, better than the 755a internal objective around `2.0605` and much better than the additive 762a objective around `2.1569`.

## Mechanism Read

Budget matching fixes the worst part of 762a: train-tail improves from `6/11` to `7/11`, coverage returns to the 755a level, level KS improves to `22/25`, and cointegration recovers above the gate. This means the 763a objective-scale diagnosis was correct.

It is still not a promotion candidate. Validation remains `6/11`, validation level KS stays below 755a (`14/25` versus `15/25`), and mean reversion regresses on both validation and train-tail full-horizon active profile. The proper level score helps support allocation but steals capacity or gradient budget from the dynamic mean-reversion profile.

## Decision

Reject 765a as the active learned base model. Keep 755a as the current best base learned IV model and keep the budgeted level-score idea as diagnostic evidence, not as a promoted recipe.

The next HEAD step should not be another nearby scalar level-score weight. The clean read is now: the current objective/readout couples level support repair against transition-shape realism. The next principled move is post-experiment analysis or ideation on a cleaner objective/readout separation that avoids adding more scalar knobs.
