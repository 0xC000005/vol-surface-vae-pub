# 278c Postmortem

## Result
- Full 11-suite score: 3/11
- Passes: surface, coverage, cross_cell_correlation

## What Was Tested
`278c` kept the same `277d` Stage A retrieval core and the same top-k future set as
`278a`, but replaced heuristic Stage B sampling with a learned query-conditioned
reweighting model over those retrieved candidates.

## Mechanism Read
- This is a clean negative.
- The learned Stage B scorer does not just add useful spread.
- It also pulls the ensemble center away from the `277d` deterministic backbone:
  - cointegration drops back below the robust local gate
  - active mean-reversion support weakens further
  - the ensemble no longer preserves the Stage A statistical-validity object well
- So the Stage B problem is now clearer:
  - the distribution layer should not choose whole future paths in a way that moves the
    center path too much
  - it should add residual spread around a fixed Stage A center path

## Decision
- Keep the two-level hierarchy.
- Keep `277d` as the fixed deterministic Stage A core.
- Close `278c` as a negative full-path weighting variant.
- Next step: `279a-v0`, residual hierarchical retrieval:
  - deterministic center path from `277d`
  - retrieved future bank converted into zero-centered residual scenarios around that
    center
