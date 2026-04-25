# 511a 392a vs 510a Frontier Comparison

## Context

510a recovered an `8/11` frontier tie from the final checkpoint of the patch
energy fine-tune. 511a compares it against the previous active frontier, 392a,
to decide which checkpoint should anchor further work.

## Comparison

| metric | 392a | 510a |
|---|---:|---:|
| score | `8/11` | `8/11` |
| failed suites | coverage, regime, fidelity | coverage, regime, fidelity |
| coverage90 | `0.8675` | `0.8732` |
| per-cell coverage min | `0.6979` | `0.7135` |
| per-cell coverage max | `0.9844` | `0.9792` |
| under-70 cells | `1` | `0` |
| over-95 cells | `10` | `13` |
| conditionality MAE reduction | `5.14%` | `5.12%` |
| cointegration worst-cell ratio | `0.278` | `0.257` |
| daily KS | `25/25` | `25/25` |
| level KS | `10/25` | `10/25` |
| median-bias cells | `20/25` | `20/25` |
| bias magnitude cells | `25/25` | `25/25` |
| regime layer2 | `0/8` | `0/8` |
| cross-cell corr ratio | `0.963` | `0.968` |
| rank ratio | `1.495` | `1.462` |
| mean-reversion ratio | `1.024` | `0.986` |
| path max-jump KS | `0.373` | `0.361` |

## Mechanism Read

510a is not strictly better. It improves the undercoverage side of the risk
profile and slightly improves path/correlation geometry, but it pays with more
over-95 cells and thinner cointegration margin. The hard scientific bottleneck
is unchanged: both have level KS `10/25` and regime layer2 `0/8`.

The useful signal is that patch-energy training moved along a nearby frontier
rather than collapsing. Since 392a and 510a are same-architecture checkpoints,
the only non-invasive follow-up is a one-shot parameter-space midpoint audit.
This is not a new objective, architecture, or calibration layer; it tests whether
the patch-energy trajectory has a better full-suite point between endpoints.

## Decision

Keep 392a as the safer structural anchor for now. Keep 510a as a risk-coverage
frontier tie. Run one midpoint checkpoint interpolation between 392a and 510a
final as 512a. If it does not exceed `8/11` or materially improve the remaining
three failures, close checkpoint interpolation immediately.
