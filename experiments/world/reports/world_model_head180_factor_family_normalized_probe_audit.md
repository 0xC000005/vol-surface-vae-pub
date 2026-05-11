# World Model HEAD180: Factor-Family Normalized Probe Audit

Date: 2026-05-11

## Iteration Type

`experiment`

## Objective Family

`downstream_probe_factor_family_normalized`; future factor-panel targets
are frozen evaluation targets only and are not Part 1 pretraining losses.

## Metric Units

`train_standardized_per_factor_family`

## Feature Probe Scaling

`train_standardized_per_feature_surface`

## Feature Surfaces

- `raw_factor_last`
- `raw_factor_flat`
- `scale_barlow_last`
- `raw_factor_last_plus_scale_barlow_last`

## Family-Normalized Summary

| family | target | raw-last norm MSE | raw-flat norm MSE | scale/best-raw | raw+scale/best-raw |
| --- | --- | ---: | ---: | ---: | ---: |
| factor_level | factor_future_mean_delta | 10402.226509 | 13871.881483 | 1.113035 | 1.003484 |
| factor_level | factor_future_range | 339.714739 | 432.244499 | 0.278097 | 1.110371 |
| factor_level | factor_future_terminal_delta | 547.702341 | 1241.369172 | 0.529314 | 0.910394 |
| factor_level | factor_future_max_abs_step | 267.415785 | 618.164910 | 0.208378 | 1.063391 |
| factor_return | factor_future_mean_delta | 150.811680 | 721.592748 | 0.544433 | 1.011094 |
| factor_return | factor_future_range | 305.178429 | 783.054426 | 0.058182 | 0.987275 |
| factor_return | factor_future_terminal_delta | 85.255096 | 181.203433 | 0.050837 | 0.894894 |
| factor_return | factor_future_max_abs_step | 205.149676 | 652.639994 | 0.040672 | 0.873100 |

## Counts

- Target-family cells: `8`.
- Scale learned standalone best-raw wins: `7`.
- Raw-factor-last plus scale improvements: `4`.
- Raw-factor-last plus scale best-raw wins: `4`.

## Caveats

- `smoke_scale_128_train_64_val`
- `targets_standardized_with_train_family_statistics`
- `still_downstream_probe_only`

## Decision

Promotion decision: `PROBE_ONLY_DO_NOT_PROMOTE`.
Probe status: `factor_family_normalized_signal_audited`.
Next step: `interpret_family_normalized_audit_against_part1_gate`.

This audit removes raw-unit target-scale dominance from the factor
probe comparison, but remains downstream evidence only. It does not
promote Part 1 or authorize Part B.
