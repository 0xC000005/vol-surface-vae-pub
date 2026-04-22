# 283c Postmortem

## Result
- Full 11-suite score: `3/11`
- Passes: `surface`, `block_ar`, `cross_cell_correlation`

## What Was Tested
`283c` kept the same `283b` anchored-support reweighting architecture and the same
weighted CRPS objective, but replaced the fully free anchored candidate support with a
fixed half-centered support around the deterministic `277d` center path.

## Mechanism Read
- The support-geometry interpolation effect is real but not enough.
- Relative to `283b`, half-centering does not produce a new regime:
  - score drops back to `3/11`
  - cointegration local robustness regresses again
  - mean-reversion active support remains far below gate
  - level-KS remains stuck at `1/25`
- So the `282b` to `283c` bracket is now clear:
  - fully fixed center preserves too much of the old support miss
  - fully free weighting gives back too much of the Stage A statistical-validity object
  - fixed half-centering does not break that tradeoff

## Decision
- Do not continue with more local centering-geometry tweaks in the `283` family.
- Next step should be post-experiment analysis across `282b`, `283a`, `283b`, and
  `283c` to decide whether the real bottleneck is now upstream in the `277d`
  support/library family itself.
