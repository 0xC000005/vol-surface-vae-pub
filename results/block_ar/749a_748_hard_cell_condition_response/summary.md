# 744a IV Hard-Cell Conditioning Response Audit

## Summary
- n_windows: `441`
- samples: `64`
- generation_time_s: `123.242`
- median_low_tertile_coverage90: `0.095238`
- median_hard_lower_miss_rate: `0.571429`
- median_abs_slope_gap: `0.056009`

## Hard Cells

| horizon | cell | cov90 | lower miss | upper miss | current shift | real slope | gen slope | width/current corr | low cov | low bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | (2,3) | 0.501 | 0.492 | 0.007 | -0.880 | 0.687 | 0.658 | 0.435 | 0.129 | 0.0212 |
| 30 | (0,2) | 0.440 | 0.549 | 0.011 | -0.702 | 0.419 | 0.472 | 0.529 | 0.279 | 0.0301 |
| 30 | (2,3) | 0.351 | 0.649 | 0.000 | -0.880 | 0.539 | 0.598 | 0.455 | 0.041 | 0.0308 |
| 30 | (3,3) | 0.406 | 0.594 | 0.000 | -0.838 | 0.634 | 0.776 | 0.699 | 0.061 | 0.0178 |

## Mechanism Read

The incumbent undercovers the shifted low-level validation region mostly through lower-tail misses. This means the model is not just too narrow globally; it is not translating the current level geometry far enough into the late-horizon lower tail for the hard cells.

## Decision

Next experiment should target generic local-geometry conditioning or robustness of the state encoder. Do not use scalar temperature or interval-score losses, because the miss is state-local and directional.
