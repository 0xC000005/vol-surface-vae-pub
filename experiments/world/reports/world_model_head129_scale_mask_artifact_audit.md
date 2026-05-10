# World Model HEAD129: Scale Mask-Artifact Audit

## Objective Family

`masked_multiview_invariance` mask-artifact diagnostic.

## Hypothesis

If HEAD127 scaled checkpoint learned market-state structure rather than corruption artifacts,
a simple frozen-embedding probe should not predict synthetic mask family
far above the majority-class baseline.

## Probe Results

| probe | accuracy | majority | lift | macro recall | classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| view_a_to_mask_family_a | 0.218750 | 0.203125 | 0.015625 | 0.233504 | 6 |
| view_b_to_mask_family_b | 0.218750 | 0.203125 | 0.015625 | 0.254847 | 6 |
| clean_to_mask_family_a | 0.203125 | 0.203125 | 0.000000 | 0.166667 | 6 |
| clean_to_mask_family_b | 0.167969 | 0.203125 | -0.035156 | 0.187179 | 6 |

## Decision

`no_large_mask_family_leakage`.

This is a diagnostic, not a training objective. A positive leakage
result should trigger mask-policy or representation-surface analysis
before any new model knobs.
