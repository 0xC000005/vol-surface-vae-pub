# World Model HEAD080: Part 1 Scorecard Consolidation

## Objective Family

`masked_multiview_invariance` scorecard consolidation.

## Hypothesis

The current Part 1 reference decision should be reproducible from saved
JSON artifacts without hand-transcribing the leaderboard.

## Masked-Multiview Leaderboard

| run | family | block | top1 | top5 | top10 | eff rank A/B | offdiag | raw top10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HEAD066 | `context_to_target_jepa` | `predicted_target` | 0.000260 | 0.002344 | 0.006510 | 4.134 / 9.048 | 0.204527 | 0.357031 |
| HEAD068 | `masked_multiview_invariance` | `view_alignment` | 0.394271 | 0.680990 | 0.853906 | 4.484 / 4.526 | 0.468407 | 0.373177 |
| HEAD070 | `masked_multiview_invariance` | `view_alignment` | 0.321354 | 0.662500 | 0.841927 | 14.501 / 14.594 | 0.216527 | 0.373177 |
| HEAD076 | `masked_multiview_invariance` | `view_alignment` | 0.138802 | 0.257292 | 0.333073 | 3.323 / 3.321 | 0.402751 | 0.373177 |

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

The next step should be a bounded metric-gap or probe-gap audit, not a
new model knob and not Part 2 decoder work.
