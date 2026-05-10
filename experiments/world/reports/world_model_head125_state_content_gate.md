# World Model HEAD125: State-Content Gate

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_present_state_information` gate.

## Verdict

- Quality gate passed: `False`.
- Promotion decision: `FAIL`.

## Layer Results

| layer | status | evidence |
| --- | --- | --- |
| non_surface_signal | PASS | factor_return_r2=0.799291; factor_level_mse=0.747878 vs raw_surface=0.922377; side_mse=0.428133 vs raw_surface=0.598214 |
| exact_iv_state_retention | FAIL | iv_surface_mse=0.014483 vs raw_surface=0.005630 |
| factor_level_gap | FAIL | factor_level_mse=0.747878 vs raw_full_geometry_upper_bound=0.113066 |
| mask_aggression_regression | FAIL | hard_all_geometry_mse=0.304687 vs default_all_geometry_mse=0.261496 |

## Interpretation

The default embedding contains useful non-surface signal, especially factor returns, but the state-content gate fails because exact IV state retention and factor-level fidelity are not strong enough.

## Next Required Evidence

- state-content probes must remain separate from future prediction losses
- improve exact IV/current-state retention without collapsing factor-return signal
- evaluate whether geometry-aware pooling or larger scale fixes the state-content gap before changing objective family
