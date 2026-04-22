# 281a Hierarchical Postmortem

## Result
- Full 11-suite score: 5/11
- Passes: surface, block_ar, cointegration, cross_cell_correlation, mean_reversion

## What Was Tested
`281a` kept the fixed `277d` Stage A center path and the same anchored residual bank,
but replaced scale-only Stage B control with a learned query-conditioned **residual
temperature** over the fixed retrieval scores. Residuals were centered under that same
weighted selection distribution so the center path stayed preserved in expectation.

## Mechanism Read
- This is a real non-scale effect.
- Relative to `280b`, temperature control improves the parts that depend on residual
  shape rather than just amplitude:
  - regime differentiation passes again
  - jump realism gets very close to the gate
- But temperature-only control is still not enough:
  - coverage remains poor
  - change-KS fidelity collapses badly again
  - distributional fidelity remains weak

So the new diagnosis is sharper:
- scale control and selection-shape control are complementary
- each one fixes part of Stage B
- neither one alone is sufficient

## Decision
- Keep the same two-level hierarchy.
- Keep the fixed Stage A center path and fixed residual bank.
- Keep residual temperature as a live mechanism class.
- Next step: `281b-v0`, the minimal joint Stage B model:
  - learned residual temperature
  - plus one learned residual scale
  - no new scorer, no new side path, no new family reset
