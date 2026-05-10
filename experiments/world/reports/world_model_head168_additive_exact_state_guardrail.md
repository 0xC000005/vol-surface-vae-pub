# World Model HEAD168: Additive Exact-State Guardrail

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_additive_signal_gate`; frozen current-state probe.

## Hypothesis

If raw exact state remains explicit, the relevant guardrail is whether
`raw_surface_plus_scale_barlow` preserves or improves raw-surface
current-state probes, not whether the learned embedding alone replaces
raw state.

## Falsifier

The additive framing would be unsafe if adding the frozen embedding to
raw current-state features materially worsens exact-state probes.

## IV Exact-State Guardrail

| feature surface | IV MSE | ratio to raw |
| --- | ---: | ---: |
| raw-only | 0.005630 | 1.000000 |
| learned-only | 0.013756 | 2.443584 |
| raw-plus-learned | 0.005901 | 1.048173 |
| raw-geometry upper | 0.006609 | n/a |

## Target Rows

| target | raw-only MSE | learned-only MSE | raw+learned MSE | raw+learned delta |
| --- | ---: | ---: | ---: | ---: |
| iv_surface | 0.005630 | 0.013756 | 0.005901 | 0.000271 |
| vol_side_channel | 0.598214 | 0.343138 | 0.353797 | -0.244418 |
| factor_level | 0.922377 | 0.663298 | 0.663217 | -0.259160 |
| factor_return | 0.785814 | 0.181501 | 0.182619 | -0.603195 |
| all_geometry | 0.466319 | 0.239427 | 0.237211 | -0.229108 |

## Decision

- Raw-plus exact-state guardrail: `FAIL`.
- Promotion decision: `DO_NOT_PROMOTE`.
- Part B blocked: `True`.

Raw-plus-learned is the correct exact-state guardrail for an additive embedding. This check does not promote Part 1 by itself; it only verifies whether adding the embedding harms or helps the explicit raw-state floor.
