# World Model HEAD179: Factor-Panel Probe Bakeoff

Date: 2026-05-11

## Iteration Type

`experiment`

## Objective Family

`downstream_probe_factor_panel_bakeoff`; future factor-panel targets
are frozen evaluation targets only and are not Part 1 pretraining losses.

## Feature Surfaces

- `raw_factor_last`
- `raw_factor_flat`
- `scale_barlow_last`
- `raw_factor_last_plus_scale_barlow_last`

## Future Factor-Panel Summary

| target | raw-last MSE | raw-flat MSE | scale/best-raw | raw+scale/raw-last |
| --- | ---: | ---: | ---: | ---: |
| factor_future_mean_delta | 12189837.571440 | 8060329.313524 | 0.001762 | 0.761535 |
| factor_future_range | 118675.704319 | 2330190.254905 | 0.058208 | 0.264541 |
| factor_future_terminal_delta | 5172739.722272 | 22206529.297915 | 0.004871 | 0.653859 |
| factor_future_max_abs_step | 381772.593977 | 891655.688889 | 0.004532 | 0.754265 |

## Counts

- Scale learned standalone best-raw wins: `4`.
- Raw-factor-last plus scale improvements: `4`.
- Raw-factor-last plus scale best-raw wins: `3`.

## Caveats

- `smoke_scale_128_train_64_val`
- `raw_unit_mse_not_column_standardized`
- `target_family_breakdown_missing`

## Decision

Promotion decision: `PROBE_ONLY_DO_NOT_PROMOTE`.
Probe status: `factor_panel_signal_present_smoke_only`.
Next step: `factor_family_normalized_probe_audit_before_promotion`.

This is downstream coverage evidence only. It does not promote Part 1
or authorize Part B.
