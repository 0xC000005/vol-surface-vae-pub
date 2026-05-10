# World Model HEAD152: Surface-Local Target Coverage

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`token_geometry_level_context_to_target_jepa_data_audit`; no model change.

## Hypothesis

The surface-local data contract should explicitly cover the wing and edge
maturity cells that dominate the HEAD149 exact-state gap.

## Coverage Summary

- Train shape: `[1024, 30, 58]`.
- Validation shape: `[256, 30, 58]`.
- Train target positions: `121348`.
- Validation target positions: `31130`.
- Validation hidden rate: `0.070007`.

## Target Families

| family | windows | target positions | hidden rate | last-row rate |
| --- | ---: | ---: | ---: | ---: |
| factor_family | 47 | 1580 | 0.019367 | 0.042553 |
| surface_atm_strip | 48 | 7200 | 0.086383 | 1.000000 |
| surface_edge_maturity | 39 | 5850 | 0.086395 | 1.000000 |
| surface_rectangle | 60 | 7200 | 0.069025 | 1.000000 |
| surface_wing_moneyness | 62 | 9300 | 0.086341 | 1.000000 |

## Surface Position Counts

| group | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| moneyness | 6630 | 2850 | 10470 | 3090 | 6510 |
| maturity | 6000 | 4740 | 5220 | 5460 | 8130 |

Top targeted IV cells:

| cell | moneyness | maturity | positions | targeted windows |
| --- | ---: | ---: | ---: | ---: |
| iv_m2_t4 | 2 | 4 | 2520 | 84 |
| iv_m2_t3 | 2 | 3 | 2130 | 71 |
| iv_m2_t2 | 2 | 2 | 2040 | 68 |
| iv_m2_t0 | 2 | 0 | 1980 | 66 |
| iv_m2_t1 | 2 | 1 | 1800 | 60 |
| iv_m4_t4 | 4 | 4 | 1800 | 60 |
| iv_m0_t4 | 0 | 4 | 1770 | 59 |
| iv_m4_t0 | 4 | 0 | 1470 | 49 |
| iv_m0_t0 | 0 | 0 | 1410 | 47 |
| iv_m0_t2 | 0 | 2 | 1170 | 39 |

## Decision

- `iv_m0_t0` target positions: `1410`.
- All IV cells targeted: `True`.
- Wing positions: `13140`.
- Core positions: `16410`.
- Edge maturity positions: `14130`.
- Middle maturity positions: `15420`.
- Covers HEAD149 problem regions: `True`.
- Promotion decision: `DATA_AUDIT_ONLY`.

The data contract covers the HEAD149 wing/edge problem regions, including iv_m0_t0, but this remains only data evidence before any encoder or loss is introduced.
