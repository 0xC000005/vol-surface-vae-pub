# 279b Hierarchical Postmortem

## Result
- Full 11-suite score: 4/11
- Passes: surface, block_ar, cointegration, cross_cell_correlation

## What Was Tested
`279b` kept the same `277d` Stage A center path and the same anchored retrieval bank as
`279a`, but only subtracted a fraction of the residual mean (`alpha = 0.5`) instead of
fully zero-centering the residual scenarios.

This was the smallest clean test of the bracket implied by:
- `278a`: enough spread, not enough center preservation
- `279a`: enough center preservation, not enough spread/fidelity preservation

## Mechanism Read
- This is a real interpolation result, not random churn.
- Relative to `279a`, `279b` recovers more of the useful spread and change-law fidelity:
  - overall coverage improves
  - change-KS passes again
  - regime width differentiation passes again
- But it gives back too much of the deterministic Stage A backbone:
  - mean-reversion falls back below the full-horizon gate
  - jump realism remains outside the gate
  - level-KS remains dead

So the current Stage B problem is clearer now:
- residual centering strength alone is not enough
- the hierarchy needs a way to control residual amplitude adaptively while preserving
  the fixed Stage A center path

## Decision
- Keep the two-level hierarchy.
- Keep `277d` as the fixed deterministic Stage A core.
- Close pure centering-strength interpolation as insufficient by itself.
- Next step: `280a-v0`, a center-preserving residual hierarchy with learned
  query-conditioned residual scaling over the fixed residual bank.
