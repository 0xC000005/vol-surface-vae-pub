# World Model HEAD122: Hard Mask Preset

Date: 2026-05-10

## Iteration Type

`research_ideation`

## Objective Family

`masked_multiview_invariance` hard-mask diagnostic design.

## Hypothesis

If HEAD070 underperforms raw/simple baselines partly because the default
masks are too mild, then a single harder structured-mask preset should
substantially reduce two-view overlap before any architecture or objective
change is attempted.

## Falsifier

The preset is not worth training if it leaves most entries visible in both
views, destroys structured market geometry, or requires many tunable knobs.

## Preset Families

- `surface_large_rectangle`
- `surface_whole_day_block`
- `factor_family_long_block`
- `cross_family_stress_block`
- `time_block_long`

## Mask Difficulty

| split | view A hidden | view B hidden | both visible | both hidden | view disagreement | union hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 23.43% | 24.12% | 59.97% | 7.53% | 32.50% | 40.03% |
| val | 22.65% | 23.60% | 60.93% | 7.18% | 31.88% | 39.07% |

Compared with HEAD070 default validation masks, which kept about `86.6%`
visible in both views, this preset creates a much harder missing-state
diagnostic while keeping semantic groups intact.

## Decision

- Use for next training smoke: `True`.
- Reason: The preset lowers view overlap enough to create a harder missing-information diagnostic while preserving structured market geometry.
- Guardrail: Use as one named preset; do not introduce per-family tuning knobs until this diagnostic is understood.
