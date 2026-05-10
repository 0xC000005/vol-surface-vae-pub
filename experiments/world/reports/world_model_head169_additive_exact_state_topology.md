# World Model HEAD169: Additive Exact-State Topology

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

## IV Cell Delta Topology

- Raw-plus worse cells: `14/25`.
- Raw-plus better cells: `11/25`.
- Mean raw-plus minus raw MSE: `0.000271`.

| cell | moneyness | maturity | raw MSE | learned MSE | raw+learned MSE | raw+learned minus raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iv_m0_t3 | 0 | 3 | 0.024818 | 0.037854 | 0.028791 | 0.003973 |
| iv_m3_t0 | 3 | 0 | 0.001089 | 0.005567 | 0.002823 | 0.001733 |
| iv_m0_t1 | 0 | 1 | 0.000804 | 0.005890 | 0.001902 | 0.001098 |
| iv_m3_t1 | 3 | 1 | 0.000347 | 0.001278 | 0.001183 | 0.000836 |
| iv_m4_t1 | 4 | 1 | 0.000181 | 0.000965 | 0.000844 | 0.000663 |
| iv_m3_t4 | 3 | 4 | 0.006739 | 0.008228 | 0.007375 | 0.000636 |
| iv_m1_t3 | 1 | 3 | 0.023544 | 0.026710 | 0.024164 | 0.000620 |
| iv_m2_t0 | 2 | 0 | 0.000428 | 0.003856 | 0.001028 | 0.000599 |
| iv_m2_t4 | 2 | 4 | 0.053792 | 0.061628 | 0.054385 | 0.000593 |
| iv_m2_t1 | 2 | 1 | 0.000413 | 0.001104 | 0.000777 | 0.000364 |

## Decision

- Raw-plus exact-state guardrail: `FAIL`.
- Promotion decision: `DO_NOT_PROMOTE`.
- Part B blocked: `True`.

Raw-plus-learned is the correct exact-state guardrail for an additive embedding. This check does not promote Part 1 by itself; it only verifies whether adding the embedding harms or helps the explicit raw-state floor.
