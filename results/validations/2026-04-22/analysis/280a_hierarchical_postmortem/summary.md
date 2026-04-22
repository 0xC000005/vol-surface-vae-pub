# 280a Hierarchical Postmortem

## Result
- Full 11-suite score: 4/11
- Passes: surface, block_ar, cointegration, cross_cell_correlation

## What Was Tested
`280a` kept the same fixed `277d` Stage A center path, the same anchored residual bank,
and the same partial residual-centering geometry as `279b`, but added a learned
query-conditioned **scalar residual scale head**.

The goal was to improve Stage B coverage/regime allocation without reintroducing center
drift, because the learned component only scales residual amplitude around the fixed
center path.

## Mechanism Read
- This is a real effect, not noise.
- Relative to `279b`, the scalar residual scale improves broad amplitude-sensitive
  metrics:
  - overall coverage improves
  - h30 coverage crosses the gate
  - coverage floor improves
  - jump KS improves
  - kurtosis ratio crosses the gate
- But the scalar scale is too blunt:
  - regime differentiation collapses back toward 1.0
  - conditionality remains weak
  - full-horizon mean-reversion still fails

So the live Stage B problem is narrower again:
- a single scale per query is not expressive enough
- the residual bank still needs **horizon-structured** scaling, not just one global
  amplitude factor

## Decision
- Keep the two-level hierarchy.
- Keep the fixed Stage A center path and fixed residual bank.
- Keep learned residual scaling as the right mechanism class.
- Close the single-scalar scale head as insufficient by itself.
- Next step: `280b-v0`, a minimal query-conditioned horizon-scale profile over the same
  residual bank.
