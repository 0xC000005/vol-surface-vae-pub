# World Model HEAD141: Context-Target State Probe

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_present_state_information` for frozen Part 1 candidates.

## Hypothesis

If the context-to-target branch is the right fix for exact-state
retention, its frozen context encoder should improve current-state probes
against scaled Barlow without collapsing representation health.

## Falsifier

The context-to-target branch is not an immediate fix if its clean context
embeddings are worse than scaled Barlow on current-IV state probes or have
materially weaker representation rank.

## Execution

Loaded the HEAD140 context-to-target smoke checkpoint and the HEAD127 scaled
Barlow checkpoint, encoded clean validation windows, and fit identical frozen
ridge probes for present-state geometry targets.

## Present-State Probe MSE

| feature | rank | IV surface | side channel | factor level | factor return | all geometry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_surface_last | 3.241518 | 0.005630 | 0.598214 | 0.922377 | 0.785814 | 0.466319 |
| raw_geometry_last_upper_bound | 11.704106 | 0.006609 | 0.108668 | 0.113066 | 0.000432 | 0.039613 |
| scale_barlow_last | 18.761132 | 0.013756 | 0.343138 | 0.663298 | 0.181501 | 0.239427 |
| context_target_last | 11.592254 | 0.015289 | 0.375932 | 0.691814 | 0.997291 | 0.446713 |

## Decision

- Context-target improves IV versus scaled Barlow: `False`.
- Context-target beats raw surface on IV: `False`.
- Context-target IV MSE: `0.015289`.
- Scaled Barlow IV MSE: `0.013756`.
- Raw surface IV MSE: `0.005630`.
- Context-target rank: `11.592254`.
- Scaled Barlow rank: `18.761132`.
- Promotion decision: `DO_NOT_PROMOTE`.

The context-to-target smoke must improve exact-state probes and retain healthy rank before it can compete with scaled Barlow.
