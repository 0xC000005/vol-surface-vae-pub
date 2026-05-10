# World Model HEAD151: Surface-Local Data Contract

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`token_geometry_level_context_to_target_jepa` data-contract scaffold.

## Hypothesis

Before any surface-local JEPA model exists, the data layer should prove it can
construct same-window token/geometry targets with explicit surface-local
families and target-token positions.

## Falsifier

The scaffold fails if target masks touch future rows, lose token geometry, fail
to hide target values from the context view, or cannot isolate wing/edge
surface targets.

## Execution

Added `experiments/world/evaluation/surface_local_jepa_data.py` and
`test_code/test_world_model_surface_local_jepa_data.py` using TDD. The test
first failed on the missing module, then passed after implementing the data
contract.

The batch exposes:

- clean values, context values, and target-only values;
- observed, context, and target masks;
- `target_positions` as `(window, relative_time, token)` rows;
- geometry token metadata for factor id/family and surface coordinates;
- per-window target family labels;
- metadata declaring `uses_future_targets=False`.

## Validation Sample

For `128` validation windows, history `30`, seed `2150`:

- shape: `[128, 30, 58]`;
- target token positions: `15184`;
- hidden rate: `0.068175`.

| target family | windows | hidden rate | target positions | last-row rate |
| --- | ---: | ---: | ---: | ---: |
| factor_family | 28 | 0.017323 | 844 | 0.000000 |
| surface_atm_strip | 25 | 0.086207 | 3750 | 1.000000 |
| surface_edge_maturity | 22 | 0.086207 | 3300 | 1.000000 |
| surface_rectangle | 22 | 0.068966 | 2640 | 1.000000 |
| surface_wing_moneyness | 31 | 0.086207 | 4650 | 1.000000 |

## Decision

The token/geometry-level data contract is ready for diagnostics. This does not
train a model, does not promote Part 1, and does not unblock Part B. The next
safe step is to audit target coverage and target-token metadata before adding
an encoder.
