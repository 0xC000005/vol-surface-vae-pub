# 744a IV Hard-Cell Conditioning Response Audit

## Summary
- n_windows: `441`
- samples: `64`
- generation_time_s: `123.312`
- median_low_tertile_coverage90: `0.289116`
- median_hard_lower_miss_rate: `0.411565`
- median_abs_slope_gap: `0.03747`

## Hard Cells

| horizon | cell | cov90 | lower miss | upper miss | current shift | real slope | gen slope | width/current corr | low cov | low bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | (2,3) | 0.676 | 0.317 | 0.007 | -0.880 | 0.687 | 0.646 | 0.316 | 0.347 | 0.0169 |
| 30 | (0,2) | 0.605 | 0.383 | 0.011 | -0.702 | 0.419 | 0.391 | 0.432 | 0.490 | 0.0261 |
| 30 | (2,3) | 0.528 | 0.472 | 0.000 | -0.880 | 0.539 | 0.482 | 0.358 | 0.088 | 0.0265 |
| 30 | (3,3) | 0.560 | 0.440 | 0.000 | -0.838 | 0.634 | 0.667 | 0.635 | 0.231 | 0.0144 |

## Mechanism Read

The incumbent undercovers the shifted low-level validation region mostly through lower-tail misses. This means the model is not just too narrow globally; it is not translating the current level geometry far enough into the late-horizon lower tail for the hard cells.

## Decision

Next experiment should target generic local-geometry conditioning or robustness of the state encoder. Do not use scalar temperature or interval-score losses, because the miss is state-local and directional.
