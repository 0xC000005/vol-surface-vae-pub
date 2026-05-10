# World Model HEAD134: Scale Baseline Target Taxonomy

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_baseline_diagnostic` for frozen scaled Part 1 features.

## Hypothesis

If the baseline-superiority failure is structured, scaled Barlow should
win or add value on path-shape/risk-width targets while losing
persistence or exact-state dominated targets.

## Target Rows

| target | family | best raw | Barlow MSE | best raw MSE | Barlow/best raw | raw last MSE | raw+Barlow MSE | combo/raw | combo improvement % | Barlow win | adds to raw last |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| future_mean_delta | persistence_or_exact_state_dominated | raw_surface_last | 0.010780 | 0.006484 | 1.662713 | 0.006484 | 0.006266 | 0.966475 | 3.352479 | False | True |
| future_range | path_shape_or_risk_width | raw_surface_flat | 0.044396 | 0.050972 | 0.871001 | 0.054625 | 0.045222 | 0.827864 | 17.213551 | True | True |
| future_terminal_delta | persistence_or_exact_state_dominated | raw_surface_last | 0.028396 | 0.019208 | 1.478311 | 0.019208 | 0.020934 | 1.089850 | -8.984981 | False | False |
| future_max_abs_step | mixed_path_shape | raw_surface_flat | 0.036666 | 0.035980 | 1.019080 | 0.041928 | 0.037278 | 0.889081 | 11.091947 | False | True |
| future_drawdown | path_shape_or_risk_width | raw_surface_flat | 0.039983 | 0.044180 | 0.904987 | 0.050058 | 0.040234 | 0.803739 | 19.626051 | True | True |

## Family Summary

| family | targets | Barlow raw wins | adds to raw last | mean Barlow/best raw | mean combo improvement % |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed_path_shape | 1 | 0 | 1 | 1.019080 | 11.091947 |
| path_shape_or_risk_width | 2 | 2 | 2 | 0.887994 | 18.419801 |
| persistence_or_exact_state_dominated | 2 | 0 | 1 | 1.570512 | -2.816251 |

## Decision

- Barlow wins path-shape/risk-width family: `True`.
- Barlow loses persistence/exact-state family: `True`.
- Promotion decision: `DO_NOT_PROMOTE`.

The scaled embedding is useful for path-shape/risk-width targets but loses persistence/exact-state dominated targets. Baseline superiority should therefore be reported by target family; the current Part 1 blocker is exact-state retention and target-family coverage, not a uniform failure of the representation.
