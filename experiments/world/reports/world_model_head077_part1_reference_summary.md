# World Model HEAD077: Part 1 Reference Summary

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Decision

HEAD070 canonical direct Barlow remains the current Part 1 reference candidate.

HEAD076 falsified the first geometry-aware alternative. The result does not
mean geometry-aware encoders are bad; it means the minimal token-descriptor plus
mean-pooling design is too lossy and should not replace HEAD070.

## Current Leaderboard

Validation masked-multiview representation metrics:

| run | top1 | top5 | top10 | effective rank A/B | offdiag abs mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| HEAD066 EMA/predictor hybrid | 0.000260 | 0.002344 | 0.006510 | 4.134 / 9.048 | 0.204527 |
| HEAD068 direct Barlow weak offdiag | 0.394271 | 0.680990 | 0.853906 | 4.484 / 4.526 | 0.468407 |
| HEAD070 canonical direct Barlow | 0.321354 | 0.662500 | 0.841927 | 14.501 / 14.594 | 0.216527 |
| HEAD076 geometry mean-pooled Barlow | 0.138802 | 0.257292 | 0.333073 | 3.323 / 3.321 | 0.402751 |

Frozen probe snapshot:

| feature | mean-delta MSE/R2 | range MSE/R2 |
| --- | ---: | ---: |
| HEAD070 clean last latent | 0.011635 / 0.162420 | 0.047185 / -2.428997 |
| raw surface last | 0.006484 / 0.533258 | 0.054625 / -2.969679 |
| raw surface last + HEAD070 | 0.006584 / 0.526010 | 0.047705 / -2.466780 |

## Interpretation

The stable finding is:

- direct two-view Barlow is the right objective family for the current
  corruption-based Part 1 objective;
- canonical off-diagonal scaling matters;
- HEAD070 gives strong same-state retrieval and healthy rank;
- HEAD070 carries useful downstream information, especially for future range;
- it does not dominate raw surface state for mean-delta;
- the first geometry-aware encoder was too lossy.

## Next Safe Step

Do not start Part 2/flow decoder work yet.

The next workflow step should be consolidation rather than another model tweak:

- freeze HEAD070 as the named Part 1 reference candidate;
- add a compact scorecard script/report that reads the saved JSON artifacts and
  reproduces the leaderboard above;
- only after the scorecard is stable, decide between:
  - a better geometry-aware architecture with non-lossy pooling/attention; or
  - downstream probe expansion around tasks where HEAD070 is already useful.

This keeps the workflow from drifting into ad hoc knobs.

## Artifacts

- `results/world/masked_multiview_barlow_head070.json`
- `results/world/masked_multiview_barlow_probe_head074.json`
- `results/world/masked_multiview_geometry_barlow_head076.json`
- `experiments/world/reports/world_model_head077_part1_reference_summary.md`
