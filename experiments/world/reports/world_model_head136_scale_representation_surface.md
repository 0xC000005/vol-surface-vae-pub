# World Model HEAD136: Scale Representation Surface

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_present_state_information` for frozen scaled Part 1 surfaces.

## Hypothesis

If the exact-state gap is mainly a readout-surface problem, then mean,
last+mean, or flattened per-time scaled embeddings should recover current
state better than the current last-state readout and possibly approach raw
last-surface baselines.

## Falsifier

If no frozen scaled representation surface beats the current last-state
readout or raw exact-state baselines, then the blocker is likely in the
learned representation/objective, not merely the probe surface.

## Present-State Probe MSE

| feature | dims | IV surface | side channel | factor level | factor return | all geometry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_surface_last | 25 | 0.005630 | 0.598214 | 0.922377 | 0.785814 | 0.466319 |
| raw_geometry_last_upper_bound | 58 | 0.006609 | 0.108668 | 0.113066 | 0.000432 | 0.039613 |
| scale_barlow_last | 64 | 0.013756 | 0.343138 | 0.663298 | 0.181501 | 0.239427 |
| scale_barlow_mean | 64 | 0.018779 | 0.637016 | 0.782042 | 0.787894 | 0.441960 |
| scale_barlow_last_plus_mean | 128 | 0.014264 | 0.359664 | 0.693162 | 0.166615 | 0.244686 |
| scale_barlow_flat_time | 1920 | 0.021086 | 0.498218 | 0.722801 | 0.188130 | 0.271919 |

## Best Scaled Surface By Target

| target | feature | MSE |
| --- | --- | ---: |
| iv_surface | scale_barlow_last | 0.013756 |
| vol_side_channel | scale_barlow_last | 0.343138 |
| factor_level | scale_barlow_last | 0.663298 |
| factor_return | scale_barlow_last_plus_mean | 0.166615 |
| all_geometry | scale_barlow_last | 0.239427 |

## Decision

- Best scaled surface beats raw surface on IV state: `False`.
- Raw surface IV MSE: `0.005630`.
- Best scaled surface IV MSE: `0.013756`.
- Promotion decision: `DO_NOT_PROMOTE`.

Changing the frozen representation readout surface does not fix the exact-state gap. Last-state embeddings remain the best scaled surface for IV, side-channel, factor-level, and all-geometry probes, while last+mean only improves factor returns. The blocker is therefore more likely in the learned representation/objective than in the downstream pooling choice.
