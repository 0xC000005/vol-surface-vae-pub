# 279a Hierarchical Postmortem

## Result
- Full 11-suite score: 5/11
- Passes: surface, block_ar, cointegration, cross_cell_correlation, mean_reversion

## What Was Tested
`279a` kept `277d` as a fixed deterministic Stage A center path and built Stage B
scenarios by:
- retrieving top-k anchored futures from the `277d` library
- converting them into residuals around the fixed center path
- subtracting the sample-mean residual
- adding the centered residuals back to the fixed center

This was the smallest clean test of whether the hierarchy can preserve the Stage A
center path while still adding useful scenario spread.

## Mechanism Read
- This is a real positive result for the two-level hierarchy.
- Relative to `278a`, `279a` preserves the Stage A deterministic backbone much better:
  - mean_reversion recovers to pass
  - cointegration remains pass
  - cross-cell structure remains pass
  - pathwise jump realism gets much closer to the gate
- But the zero-centering step is too strong:
  - coverage falls materially below `278a`
  - change-KS fidelity collapses from the strong `278a` regime
  - the distribution layer is now too centered and too blunt

The clean interpretation is:
- `278a` and `279a` form a useful bracket
- `278a` = enough spread, not enough center preservation
- `279a` = enough center preservation, not enough spread/fidelity preservation

So the live Stage B problem is now narrow:
- preserve the `277d` deterministic center path
- while only partially removing the residual mean, instead of forcing full zero-centered
  residual scenarios

## Decision
- Keep the two-level hierarchy.
- Keep `277d` as the fixed Stage A core.
- Keep retrieval-based Stage B.
- Close `279a` as the proof that center-preserving residualization is directionally right.
- Next step: `279b-v0`, partial residual centering:
  - interpolate between raw anchored retrieved futures (`278a`) and fully centered
    residual scenarios (`279a`)
  - test whether partial centering preserves enough spread while keeping the Stage A
    center path statistically valid
