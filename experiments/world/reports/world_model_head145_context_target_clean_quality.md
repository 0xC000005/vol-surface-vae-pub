# World Model HEAD145: Context-Target Clean Quality

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_present_state_information` for frozen Part 1 candidates.

## Hypothesis

If the clean-target correction fixed the HEAD140 target artifact, its clean
context encoder should improve exact-state probes and rank against the
target-only branch and scaled Barlow.

## Falsifier

The correction is insufficient if clean-target embeddings remain worse than
target-only or scaled Barlow on current-IV probes and representation rank.

## Present-State Probe MSE

| feature | rank | IV surface | side channel | factor level | factor return | all geometry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_surface_last | 3.241518 | 0.005630 | 0.598214 | 0.922377 | 0.785814 | 0.466319 |
| raw_geometry_last_upper_bound | 11.704106 | 0.006609 | 0.108668 | 0.113066 | 0.000432 | 0.039613 |
| scale_barlow_last | 18.761132 | 0.013756 | 0.343138 | 0.663298 | 0.181501 | 0.239427 |
| target_only_head140_last | 11.592254 | 0.015289 | 0.375932 | 0.691814 | 0.997291 | 0.446713 |
| clean_target_head144_last | 8.059559 | 0.015907 | 0.481423 | 0.631477 | 0.321242 | 0.278325 |

## Decision

- Clean-target improves IV versus target-only: `False`.
- Clean-target improves IV versus scaled Barlow: `False`.
- Clean-target beats raw surface on IV: `False`.
- Clean-target IV MSE: `0.015907`.
- Target-only IV MSE: `0.015289`.
- Scaled Barlow IV MSE: `0.013756`.
- Raw surface IV MSE: `0.005630`.
- Clean-target rank: `8.059559`.
- Target-only rank: `11.592254`.
- Scaled Barlow rank: `18.761132`.
- Promotion decision: `DO_NOT_PROMOTE`.

The clean-target correction must beat target-only and scaled Barlow on exact-state probes and rank before this branch can continue.
