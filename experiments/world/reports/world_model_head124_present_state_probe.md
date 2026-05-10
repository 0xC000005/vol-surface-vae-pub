# World Model HEAD124: Present-State Probe

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_present_state_information` for frozen Part 1 embeddings.

## Hypothesis

A useful market-state representation should retain enough present-state
geometry that linear probes can recover IV-surface, side-channel, and
factor-panel values better than a constant baseline and, for non-surface
targets, better than raw IV-surface-only features.

## Falsifier

If Barlow embeddings cannot recover factor-panel or side-channel state,
then Part 1 is not yet a certified joint market-state representation,
even if same-state retrieval is non-collapsed.

## Present-State Probe MSE

| feature | target | MSE | R2 | effective rank |
| --- | --- | ---: | ---: | ---: |
| head070_default_barlow_last | iv_surface | 0.014483 | -0.061864 | 12.795929 |
| head070_default_barlow_last | vol_side_channel | 0.428133 | -1.043114 | 12.795929 |
| head070_default_barlow_last | factor_level | 0.747878 | -12.241228 | 12.795929 |
| head070_default_barlow_last | factor_return | 0.156695 | 0.799291 | 12.795929 |
| head070_default_barlow_last | all_geometry | 0.261496 | -0.156943 | 12.795929 |
| head123_hard_barlow_last | iv_surface | 0.020132 | -0.476045 | 8.881722 |
| head123_hard_barlow_last | vol_side_channel | 0.407004 | -0.942281 | 8.881722 |
| head123_hard_barlow_last | factor_level | 0.874507 | -14.483202 | 8.881722 |
| head123_hard_barlow_last | factor_return | 0.206457 | 0.735550 | 8.881722 |
| head123_hard_barlow_last | all_geometry | 0.304687 | -0.348033 | 8.881722 |
| raw_surface_last | iv_surface | 0.005630 | 0.587258 | 3.241518 |
| raw_surface_last | vol_side_channel | 0.598214 | -1.854767 | 3.241518 |
| raw_surface_last | factor_level | 0.922377 | -15.330731 | 3.241518 |
| raw_surface_last | factor_return | 0.785814 | -0.006542 | 3.241518 |
| raw_surface_last | all_geometry | 0.466319 | -1.063144 | 3.241518 |
| raw_geometry_last_upper_bound | iv_surface | 0.006609 | 0.515472 | 11.704106 |
| raw_geometry_last_upper_bound | vol_side_channel | 0.108668 | 0.481421 | 11.704106 |
| raw_geometry_last_upper_bound | factor_level | 0.113066 | -1.001844 | 11.704106 |
| raw_geometry_last_upper_bound | factor_return | 0.000432 | 0.999446 | 11.704106 |
| raw_geometry_last_upper_bound | all_geometry | 0.039613 | 0.824741 | 11.704106 |

## Constant Baseline

| target | MSE | R2 |
| --- | ---: | ---: |
| iv_surface | 0.026234 | -0.923385 |
| vol_side_channel | 0.799690 | -2.816237 |
| factor_level | 1.256292 | -21.242711 |
| factor_return | 0.785090 | -0.005615 |
| all_geometry | 0.572994 | -1.535109 |

## Interpretation

The default Barlow embedding is not empty and is not ignoring the
factor panel entirely. It is much better than raw IV-surface-only
features for `factor_return`, and it is better than raw surface-only
features and the constant baseline on `factor_level` and
`vol_side_channel` MSE.

The failure is more specific: it compresses away too much exact
present-state geometry. It is worse than raw IV-surface-only features
on the IV surface itself, weak on factor levels and side channels, and
far from the raw full-geometry upper bound. The hard-mask checkpoint
does not fix this; it generally lowers rank and worsens present-state
probe quality except for a small side-channel MSE improvement.

This helps explain why simple market-state baselines win persistence-like
future probes. Those baselines carry exact current IV levels, while the
Barlow embedding is optimized for masked-view invariance and compresses
state details that those probes reward.

## Decision

This is a frozen present-state information audit, not a pretraining
objective. Use it to decide whether the representation is actually
encoding the joint market state before changing mask strength or
architecture.
