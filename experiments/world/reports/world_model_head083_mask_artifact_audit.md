# World Model HEAD083: Mask-Artifact Leakage Audit

## Objective Family

`masked_multiview_invariance` mask-artifact diagnostic.

## Hypothesis

If HEAD070 learned market-state structure rather than corruption artifacts,
a simple frozen-embedding probe should not predict synthetic mask family
far above the majority-class baseline.

## Probe Results

| probe | accuracy | majority | lift | macro recall | classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| view_a_to_mask_family_a | 0.179688 | 0.210938 | -0.031250 | 0.210042 | 6 |
| view_b_to_mask_family_b | 0.203125 | 0.257812 | -0.054688 | 0.236291 | 6 |
| clean_to_mask_family_a | 0.132812 | 0.210938 | -0.078125 | 0.166667 | 6 |
| clean_to_mask_family_b | 0.125000 | 0.257812 | -0.132812 | 0.146254 | 6 |

## Decision

`no_large_mask_family_leakage`.

This is a diagnostic, not a training objective. A positive leakage
result should trigger mask-policy or representation-surface analysis
before any new model knobs.
