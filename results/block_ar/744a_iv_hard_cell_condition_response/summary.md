# 744a IV Hard-Cell Conditioning Response Audit

## Summary
- n_windows: `441`
- samples: `64`
- generation_time_s: `123.375`
- median_low_tertile_coverage90: `0.316327`
- median_hard_lower_miss_rate: `0.394558`
- median_abs_slope_gap: `0.057789`

## Hard Cells

| horizon | cell | cov90 | lower miss | upper miss | current shift | real slope | gen slope | width/current corr | low cov | low bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | (2,3) | 0.667 | 0.329 | 0.005 | -0.880 | 0.687 | 0.698 | 0.356 | 0.347 | 0.0176 |
| 30 | (0,2) | 0.628 | 0.365 | 0.007 | -0.702 | 0.419 | 0.492 | 0.514 | 0.551 | 0.0251 |
| 30 | (2,3) | 0.519 | 0.481 | 0.000 | -0.880 | 0.539 | 0.582 | 0.388 | 0.109 | 0.0275 |
| 30 | (3,3) | 0.576 | 0.424 | 0.000 | -0.838 | 0.634 | 0.755 | 0.645 | 0.286 | 0.0144 |

## Mechanism Read

The incumbent undercovers the shifted low-level validation region mostly through lower-tail misses. This means the model is not just too narrow globally; it is not translating the current level geometry far enough into the late-horizon lower tail for the hard cells.

## Decision

Next experiment should target generic local-geometry conditioning or robustness of the state encoder. Do not use scalar temperature or interval-score losses, because the miss is state-local and directional.
