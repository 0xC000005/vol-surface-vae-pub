# World Model HEAD132: Scale Exact-State Gap

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_present_state_information` for the frozen scaled Part 1 embedding.

## Hypothesis

If the remaining raw-baseline gap is an exact-state retention problem,
the scaled embedding should be worse than raw last-surface features on
current IV-surface reconstruction even though it remains healthy under
retrieval, rank, and mask audits.

## Falsifier

The exact-state gap would be weaker if the scaled embedding matched or
beat raw last-surface features on most IV cells or if the gap were only
concentrated in one narrow surface region.

## Overall Present-State Probe MSE

| feature | IV surface | side channel | factor level | factor return | all geometry |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_surface | 0.005630 | 0.598214 | 0.922377 | 0.785814 | 0.466319 |
| raw_geometry_upper | 0.006609 | 0.108668 | 0.113066 | 0.000432 | 0.039613 |
| scale_barlow | 0.013756 | 0.343138 | 0.663298 | 0.181501 | 0.239427 |

## IV Surface Gap

- Raw last-surface IV MSE: `0.005630`.
- Scaled Barlow IV MSE: `0.013756`.
- Scaled/raw IV MSE ratio: `2.443584`.
- Cells where scaled Barlow is worse than raw surface: `20/25`.

Largest cellwise gaps:

| cell | moneyness | maturity | raw MSE | scale MSE | scale-raw | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iv_m0_t0 | 0 | 0 | 0.001537 | 0.124912 | 0.123374 | 81.251088 |
| iv_m0_t4 | 0 | 4 | 0.001011 | 0.016141 | 0.015130 | 15.964052 |
| iv_m1_t4 | 1 | 4 | 0.001583 | 0.015149 | 0.013566 | 9.572616 |
| iv_m0_t3 | 0 | 3 | 0.024818 | 0.037854 | 0.013036 | 1.525287 |
| iv_m1_t0 | 1 | 0 | 0.000257 | 0.009975 | 0.009717 | 38.748708 |
| iv_m2_t4 | 2 | 4 | 0.053792 | 0.061628 | 0.007836 | 1.145667 |
| iv_m0_t1 | 0 | 1 | 0.000804 | 0.005890 | 0.005085 | 7.322765 |
| iv_m3_t0 | 3 | 0 | 0.001089 | 0.005567 | 0.004478 | 5.110623 |
| iv_m2_t0 | 2 | 0 | 0.000428 | 0.003856 | 0.003428 | 9.007297 |
| iv_m1_t3 | 1 | 3 | 0.023544 | 0.026710 | 0.003165 | 1.134435 |

## Factor-Level MSE By Family

| family | tokens | raw surface | raw geometry upper | scale Barlow |
| --- | ---: | ---: | ---: | ---: |
| rates | 2 | 2.635379 | 0.204821 | 1.626963 |
| equity_risk | 3 | 0.687488 | 0.228282 | 0.760632 |
| commodity | 4 | 1.141227 | 0.099566 | 0.750350 |
| fx | 3 | 0.220046 | 0.022226 | 0.186823 |
| credit | 2 | 0.177502 | 0.011748 | 0.094240 |

## Decision

- Exact IV retention gap confirmed: `True`.
- Scaled Barlow worse than raw surface on all IV cells: `False`.

The scaled embedding retains broad market-state signal, but exact IV surface reconstruction remains worse than the raw last-surface baseline across most of the surface grid. This supports treating the next blocker as exact-state retention/baseline certification rather than representation collapse or seed instability.
