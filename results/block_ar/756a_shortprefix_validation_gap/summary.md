# 756a Short-Prefix Validation Gap

## Scorecards

| run | pass | cov90 | calerr | daily KS | level KS | bias | coint | worst coint | regime L2 | risk | MR | path KS | kurt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 734a_val_incumbent | 6/11 | 0.836 | 0.046 | 24/25 | 17/25 | 16/25 | 0.723 | 0.209 | 0/8 | True | True | 0.442 | 1.003 |
| 746a_train_tail_incumbent | 6/11 | 0.861 | 0.014 | 25/25 | 20/25 | 25/25 | 0.492 | 0.276 | 2/8 | False | True | 0.529 | 1.783 |
| 755a_val_shortprefix | 6/11 | 0.817 | 0.063 | 24/25 | 15/25 | 15/25 | 0.746 | 0.239 | 0/8 | True | True | 0.395 | 1.099 |
| 755a_train_tail_shortprefix | 8/11 | 0.843 | 0.031 | 25/25 | 21/25 | 25/25 | 0.550 | 0.279 | 1/8 | False | True | 0.490 | 1.943 |

## Hard-Cell Audit

| run | low-tertile cov90 | lower-miss rate | abs slope gap |
|---|---:|---:|---:|
| 744a_incumbent | 0.316 | 0.395 | 0.058 |
| 756a_755a | 0.289 | 0.412 | 0.037 |

## Attribution

- primary read: `shortprefix_repairs_in_sample_path_law_but_not_validation_shifted_hard_cells`
- 755a train-tail reaches 8/11 and passes coverage, level distribution, median bias, cointegration, mean reversion, and pathwise realism.
- 755a validation stays 6/11 and still fails coverage, conditionality, cointegration worst-cell, regime coverage, and median-bias distributional fidelity.
- Validation hard-cell low-tertile coverage remains below incumbent: 0.289116 versus 0.316327.
- Validation hard-cell lower-miss remains slightly worse than incumbent: 0.411565 versus 0.394558.

## Failure Classification

- `core_path_realism`: `mostly_repaired_in_sample`
- `validation_level_support`: `still_binding`
- `validation_conditionality`: `broad_risk_state_passes_but_per_cell_width_gate_fails`
- `regime_layer2`: `still_binding_due_per_cell_regime_coverage_not_catastrophic_undercoverage`
- `pure_capacity_or_full_prefix_loss`: `not_supported_by_755a`

## Decision

Do not abandon short-prefix exposure. The next move should target validation level-support allocation, not generic path realism. A clean next experiment should either add a schedule around the short-prefix exposure or a split-robust, data-derived support/quantile calibration inside the same normalized-innovation law; avoid broad scalar temperature tuning.
