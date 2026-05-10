# World Model HEAD156: Surface-Local Target Geometry Audit

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`token_geometry_level_context_to_target_jepa` geometry/family diagnosis.

## Hypothesis

HEAD154/155 may be failing because selected target-token latents encode
coarse geometry or family identity more readily than exact market-state rows.

## Predictor To Target Top-K

- Diagnostic rows: `512`.
- Diagnostic subset: `evenly_spaced_across_target_positions`.
- Top-k: `10`.
- Exact row any-topk: `0.042969`.
- Exact row neighbor share: `0.004297`.

| label | any top-k | neighbor share | random neighbor share |
| --- | ---: | ---: | ---: |
| factor_id | 0.187500 | 0.045703 | 0.046095 |
| geometry_id | 1.000000 | 0.906055 | 0.832215 |
| factor_family | 0.945312 | 0.875781 | 0.829960 |
| target_family | 0.693359 | 0.337891 | 0.227044 |
| relative_time | 0.238281 | 0.038281 | 0.040423 |
| window | 0.193359 | 0.028711 | 0.017712 |

## Target Intrinsic Top-K

| label | any top-k | neighbor share | random neighbor share |
| --- | ---: | ---: | ---: |
| factor_id | 0.937500 | 0.485938 | 0.046095 |
| geometry_id | 1.000000 | 0.974414 | 0.832215 |
| factor_family | 1.000000 | 0.960156 | 0.829960 |
| target_family | 0.982422 | 0.537891 | 0.227044 |
| relative_time | 0.814453 | 0.290039 | 0.040423 |
| window | 0.259766 | 0.034570 | 0.017712 |

## Decision

Promotion decision: `DO_NOT_PROMOTE`.

- Predictor retrieves factor more than exact row: `True`.
- Target factor neighbor over random: `10.542023`.
- Target family neighbor over random: `2.369102`.
- Predictor target-family neighbor over random: `1.488216`.
- Target latent factor dominated: `True`.

The clean target latent is strongly organized by token/factor and target-family labels, while predictor-to-target retrieval mostly recovers coarse geometry/family structure rather than exact rows. This points to representation geometry, not target coverage, as the next failure layer.

Next: Check whether the surface-local objective needs a target latent surface with stronger state variation before tuning model size or mask policy.
