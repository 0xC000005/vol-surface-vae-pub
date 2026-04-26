# 554a Factor Failure-Signal Audit

## Question

Do observed non-IV factor histories predict where the current IV-only frontier actually under-includes future scenarios?

## Setup

- model: `340c`
- checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- calibration windows: `192`
- validation windows: `192`
- samples per window: `32`

## Frontier Failure Geometry

- validation mean 90% coverage: `0.859`
- validation mean upper stress miss rate: `0.054`
- validation mean lower miss rate: `0.087`

## Factor Failure Signal

- material factor failure signal: `True`
- best failure abs Spearman: `0.534`
- best failure row: `{'feature': 'levels_hist_mean', 'target': 'coverage_under_target', 'spearman': 0.533993327633791, 'abs_spearman': 0.533993327633791}`
- OOS factor score Spearman to undercoverage: `-0.108`
- OOS factor score high-low undercoverage lift: `-0.0116`
- OOS IV-only score Spearman to undercoverage: `0.146`
- OOS IV-only score high-low undercoverage lift: `0.0250`

## Top Factor Associations

- `price_hist_mean` vs `gt_path_max_jump`: Spearman `-0.661`
- `levels_hist_mean` vs `gt_path_max_jump`: Spearman `0.644`
- `slopes_hist_mean` vs `gt_path_max_jump`: Spearman `-0.639`
- `price_hist_std` vs `gt_path_max_jump`: Spearman `0.586`
- `price_hist_abs_move_mean` vs `gt_path_max_jump`: Spearman `0.567`
- `price_hist_abs_q90` vs `gt_path_max_jump`: Spearman `0.565`
- `skews_hist_mean` vs `gt_path_max_jump`: Spearman `-0.564`
- `levels_hist_abs_move_mean` vs `gt_path_max_jump`: Spearman `0.558`
- `ret_hist_abs_move_mean` vs `gt_path_max_jump`: Spearman `0.557`
- `price_hist_last` vs `gt_path_max_jump`: Spearman `-0.544`
- `levels_hist_mean` vs `coverage_under_target`: Spearman `0.534`
- `ret_hist_std` vs `gt_path_max_jump`: Spearman `0.530`

## Top IV-History Associations

- `iv_hist_abs_move_q90` vs `gt_path_max_jump`: Spearman `0.609`
- `iv_hist_abs_move_mean` vs `future_abs_move_mean`: Spearman `-0.456`
- `iv_hist_trend_mean` vs `future_abs_move_mean`: Spearman `0.432`
- `iv_hist_abs_move_q90` vs `future_abs_move_mean`: Spearman `-0.425`
- `iv_hist_abs_move_mean` vs `upper_miss_rate`: Spearman `-0.336`
- `iv_hist_abs_move_mean` vs `gt_path_max_jump`: Spearman `0.297`
- `iv_hist_abs_move_q90` vs `coverage_under_target`: Spearman `0.268`
- `iv_hist_trend_mean` vs `gt_path_max_jump`: Spearman `-0.265`

## Decision

Factor history has validation failure signal, but the pre-validation ridge score does not cleanly dominate IV-history scoring out of sample. The next step should train a factor-conditioned core rather than rely on post-hoc factor calibration.
