# World Model HEAD082: Part 1 Scorecard Health Expansion

## Objective Family

`masked_multiview_invariance` scorecard consolidation.

## Hypothesis

The current Part 1 reference decision should be reproducible from saved
JSON artifacts without hand-transcribing the leaderboard.

## Masked-Multiview Leaderboard

| run | family | block | top1 | top5 | top10 | eff rank A/B | sv top1 A/B | health offdiag A/B | raw top10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HEAD066 | `context_to_target_jepa` | `predicted_target` | 0.000260 | 0.002344 | 0.006510 | 4.134 / 9.048 | 0.286 / 0.154 | 0.390 / 0.228 | 0.357031 |
| HEAD068 | `masked_multiview_invariance` | `view_alignment` | 0.394271 | 0.680990 | 0.853906 | 4.484 / 4.526 | 0.228 / 0.227 | 0.473 / 0.471 | 0.373177 |
| HEAD070 | `masked_multiview_invariance` | `view_alignment` | 0.321354 | 0.662500 | 0.841927 | 14.501 / 14.594 | 0.091 / 0.090 | 0.224 / 0.223 | 0.373177 |
| HEAD076 | `masked_multiview_invariance` | `view_alignment` | 0.138802 | 0.257292 | 0.333073 | 3.323 / 3.321 | 0.330 / 0.329 | 0.440 / 0.439 | 0.373177 |

## Health Detail

| run | variance min A/B | variance max A/B | sv top4 A/B | Barlow offdiag |
| --- | ---: | ---: | ---: | ---: |
| HEAD066 | 0.000287 / 0.015142 | 0.003550 / 0.044792 | 0.714 / 0.482 | 0.204527 |
| HEAD068 | 0.008337 / 0.008676 | 0.215665 / 0.215734 | 0.468 / 0.466 | 0.468407 |
| HEAD070 | 0.020293 / 0.020221 | 0.139303 / 0.136103 | 0.301 / 0.301 | 0.216527 |
| HEAD076 | 0.000616 / 0.000613 | 0.054721 / 0.054476 | 0.821 / 0.818 | 0.402751 |

## Frozen Probe Snapshot

| feature | mean-delta MSE/R2 | range MSE/R2 |
| --- | ---: | ---: |
| barlow_clean_last | 0.011635 / 0.162420 | 0.047185 / -2.428997 |
| raw_surface_last | 0.006484 / 0.533258 | 0.054625 / -2.969679 |
| raw_surface_last_plus_barlow_clean_last | 0.006584 / 0.526010 | 0.047705 / -2.466780 |

## Decision

HEAD070 remains the Part 1 reference candidate: it preserves strong
same-state retrieval while repairing the low-rank failure seen in
HEAD068 and avoiding the EMA/predictor collapse seen in HEAD066.

The next step should be a bounded mask-artifact or geometry-stratified
audit, not a new model knob and not Part 2 decoder work.
