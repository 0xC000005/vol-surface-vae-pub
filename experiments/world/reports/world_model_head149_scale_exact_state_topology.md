# World Model HEAD149: Scale Exact-State Topology

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_present_state_information`; no model change.

## Hypothesis

If the exact-state blocker is a surface-geometry issue, the IV gap should
show structure across moneyness, maturity, wings, or edge maturities rather
than being a single-cell artifact.

## Overall IV Gap

- Raw last-surface IV MSE: `0.005630`.
- Scaled Barlow IV MSE: `0.013756`.
- Scaled/raw ratio: `2.443584`.
- Cells where scaled Barlow is worse: `20/25`.

## By Moneyness

| group | cells | raw MSE | scale MSE | scale-raw | ratio | worse cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5 | 0.006225 | 0.037216 | 0.030991 | 5.978615 | 4 |
| 1 | 5 | 0.005542 | 0.010759 | 0.005217 | 1.941430 | 4 |
| 2 | 5 | 0.013004 | 0.015543 | 0.002538 | 1.195196 | 4 |
| 3 | 5 | 0.002436 | 0.003862 | 0.001426 | 1.585214 | 4 |
| 4 | 5 | 0.000940 | 0.001401 | 0.000461 | 1.490369 | 4 |

## By Maturity

| group | cells | raw MSE | scale MSE | scale-raw | ratio | worse cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5 | 0.000713 | 0.029189 | 0.028475 | 40.932895 | 5 |
| 1 | 5 | 0.000478 | 0.002033 | 0.001555 | 4.254729 | 5 |
| 2 | 5 | 0.001183 | 0.000670 | -0.000512 | 0.566760 | 1 |
| 3 | 5 | 0.012552 | 0.016023 | 0.003471 | 1.276501 | 4 |
| 4 | 5 | 0.013222 | 0.020867 | 0.007645 | 1.578186 | 5 |

## Wing/Core

| group | cells | raw MSE | scale MSE | scale-raw | ratio | worse cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wing_moneyness | 10 | 0.003582 | 0.019309 | 0.015726 | 5.389721 | 8 |
| core_moneyness | 15 | 0.006994 | 0.010055 | 0.003061 | 1.437576 | 12 |

## Edge/Middle Maturity

| group | cells | raw MSE | scale MSE | scale-raw | ratio | worse cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| edge_maturity | 10 | 0.006968 | 0.025028 | 0.018060 | 3.592037 | 10 |
| middle_maturity | 15 | 0.004738 | 0.006242 | 0.001504 | 1.317565 | 10 |

## Decision

- Gap broad across surface: `True`.
- Wing gap larger than core: `True`.
- Edge maturity gap larger than middle: `True`.
- Largest gap cell: `iv_m0_t0`.
- Promotion decision: `DO_NOT_PROMOTE`.

The exact-state gap is broad enough to block promotion, but it is especially concentrated in wing moneyness and edge maturities. A future design should target surface-local geometry rather than global row-level objectives.
