# World Model HEAD126: Grouped Geometry State Probe

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`masked_multiview_invariance_architecture_diagnostic`.

## Literature Status

`supported_adjacent_direct_barlow_with_grouped_geometry_encoder`.

## Hypothesis

A grouped geometry encoder should improve present-state content by keeping
IV surface, side channels, factor levels, and factor returns separable
before temporal fusion, while retaining the same masked-multiview Barlow
objective.

## Falsifier

The architecture is not justified if it improves retrieval by becoming
lower-rank, fails to improve current-state probes, or still loses exact
IV-state retention to raw surface features.

## Representation Metrics

| run | top1 | top10 | mrr | median rank | raw top10 | effective rank | variance min | offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head070_flat | 0.321354 | 0.841927 | 0.478177 | 3.000000 | 0.373177 | 14.501471 | 0.020293 | 0.224284 |
| head126_grouped | 0.512240 | 0.846094 | 0.628952 | 1.000000 | 0.379427 | 7.887283 | 0.001952 | 0.282557 |

## Present-State Probe MSE

| feature | target | MSE | R2 | effective rank |
| --- | --- | ---: | ---: | ---: |
| head070_flat_barlow_last | iv_surface | 0.014483 | -0.061864 | 12.795929 |
| head070_flat_barlow_last | vol_side_channel | 0.428133 | -1.043114 | 12.795929 |
| head070_flat_barlow_last | factor_level | 0.747878 | -12.241228 | 12.795929 |
| head070_flat_barlow_last | factor_return | 0.156695 | 0.799291 | 12.795929 |
| head070_flat_barlow_last | all_geometry | 0.261496 | -0.156943 | 12.795929 |
| head126_grouped_barlow_last | iv_surface | 0.020111 | -0.474459 | 5.021124 |
| head126_grouped_barlow_last | vol_side_channel | 0.466090 | -1.224253 | 5.021124 |
| head126_grouped_barlow_last | factor_level | 0.973286 | -16.232082 | 5.021124 |
| head126_grouped_barlow_last | factor_return | 0.676172 | 0.133897 | 5.021124 |
| head126_grouped_barlow_last | all_geometry | 0.446994 | -0.977644 | 5.021124 |
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

## Decision

- Grouped improves factor-level MSE: `False`.
- Grouped improves IV-surface MSE: `False`.
- Grouped beats raw surface on IV state: `False`.
- Grouped retrieval not worse: `True`.
- Grouped rank not worse: `False`.
- Promote grouped geometry: `False`.
