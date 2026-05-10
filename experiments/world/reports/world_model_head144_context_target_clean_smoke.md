# World Model HEAD144: Context-Target Clean-Target Smoke

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`context_to_target_jepa`.

## Literature Status

`canonical_jepa_clean_target_output_selection`.

## Hypothesis

If HEAD140 failed because the target encoder saw target-only mask artifacts, a
clean-target encoder input should improve clean context representation health
while preserving the same same-window, no-future-target objective.

## Falsifier

Reject this correction as an immediate Part 1 improvement if clean context rank
falls below HEAD140 or remains far below the scaled Barlow candidate.

## Execution

Added `context_target_jepa_clean_target_smoke.py` and a focused TDD test. The
new branch feeds the target encoder clean full-window values with observed-mask
channels, then selects target rows from the target-encoder output. It does not
use future targets, value reconstruction, decoder losses, or a knob sweep.

Ran:

```bash
python experiments/world/part1_jepa_latent/context_target_jepa_clean_target_smoke.py --device cpu --epochs 8 --max_train_windows 384 --max_val_windows 128 --seed 2144 --output_json results/world/context_target_jepa_clean_head144.json --checkpoint models/world/checkpoints/part1_jepa_latent/context_target_jepa_clean_head144.pt
```

## Result

| candidate | train loss first | train loss last | val loss | val alignment | clean last rank | variance min | offdiag abs mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HEAD144 clean-target | 0.297109 | 0.017840 | 0.077226 | 0.066922 | 7.322453 | 0.008521 | 0.340057 |
| HEAD140 target-only | 0.291315 | 0.027290 | 0.036270 | 0.025615 | 9.414709 | 0.002395 | 0.303647 |
| HEAD127 scaled Barlow | n/a | n/a | n/a | n/a | 22.323718 | 0.021119 | 0.168158 |

The clean-target correction removes the target-only input artifact by design,
but it does not improve representation health. Clean last-rank is worse than
HEAD140 and much worse than scaled Barlow.

## Decision

`DO_NOT_PROMOTE`. The canonical clean-target correction by itself does not fix
Part 1. Next, run one exact-state/latent-health comparison for the clean-target
checkpoint to decide whether this context-to-target route should be demoted or
requires a deeper architectural change rather than a knob tweak.
