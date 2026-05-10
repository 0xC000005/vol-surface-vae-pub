# World Model HEAD127: Scale State Probe

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`masked_multiview_invariance_scale_diagnostic`.

## Literature Status

`same_objective_scale_and_stability_diagnostic`.

## Hypothesis

If the state-content gap is partly a smoke-scale issue, then training the
same HEAD070-style flat encoder with more windows should improve
effective rank and present-state probes without adding future targets or
new objective terms.

## Falsifier

Scale is not enough if the checkpoint improves retrieval/rank but still
fails exact IV retention, factor-level retention, or factor-return signal.

## Representation Metrics

| run | train windows | val windows | top1 | top10 | mrr | median rank | raw top10 | effective rank | variance min | offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head070_smoke | 384.000000 | 128.000000 | 0.321354 | 0.841927 | 0.478177 | 3.000000 | 0.373177 | 14.501471 | 0.020293 | 0.224284 |
| head127_scale | 1024.000000 | 256.000000 | 0.353776 | 0.849740 | 0.497244 | 3.000000 | 0.343229 | 22.323718 | 0.021119 | 0.168158 |

## Present-State Probe MSE

| feature | target | MSE | R2 | effective rank |
| --- | --- | ---: | ---: | ---: |
| head070_smoke_barlow_last | iv_surface | 0.014483 | -0.061864 | 12.795929 |
| head070_smoke_barlow_last | vol_side_channel | 0.428133 | -1.043114 | 12.795929 |
| head070_smoke_barlow_last | factor_level | 0.747878 | -12.241228 | 12.795929 |
| head070_smoke_barlow_last | factor_return | 0.156695 | 0.799291 | 12.795929 |
| head070_smoke_barlow_last | all_geometry | 0.261496 | -0.156943 | 12.795929 |
| head127_scale_barlow_last | iv_surface | 0.013756 | -0.008570 | 18.761132 |
| head127_scale_barlow_last | vol_side_channel | 0.343138 | -0.637506 | 18.761132 |
| head127_scale_barlow_last | factor_level | 0.663298 | -10.743731 | 18.761132 |
| head127_scale_barlow_last | factor_return | 0.181501 | 0.767517 | 18.761132 |
| head127_scale_barlow_last | all_geometry | 0.239427 | -0.059303 | 18.761132 |
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

- Scale improves IV-surface MSE: `True`.
- Scale beats raw surface on IV state: `False`.
- Scale improves factor-level MSE: `True`.
- Scale preserves factor-return signal: `True`.
- Scale rank improves: `True`.
- Scale retrieval improves: `True`.
- Promote scale checkpoint: `True`.
- Promotion scope: `candidate_for_next_quality_gate_not_part_b_ready`.

The scale checkpoint is a better Part 1 candidate than HEAD070 on
this diagnostic, but it still does not beat raw surface features on
exact current-IV reconstruction. Treat it as the next candidate to
gate, not as Part-B-ready evidence.
