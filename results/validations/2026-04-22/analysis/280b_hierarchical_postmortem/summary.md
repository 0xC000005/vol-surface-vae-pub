# 280b Hierarchical Postmortem

## Result
- Full 11-suite score: 5/11
- Passes: surface, block_ar, cointegration, cross_cell_correlation, mean_reversion

## What Was Tested
`280b` kept the same fixed `277d` Stage A center path, the same anchored residual bank,
and the same partial residual-centering geometry, but replaced the scalar residual
scale head from `280a` with a tiny query-conditioned **4-knot horizon scale profile**
interpolated across the 30-day horizon.

## Mechanism Read
- This is a real structural effect.
- Relative to `280a`, horizon-structured residual scaling restores the full-horizon
  mean-reversion pass and keeps the deterministic Stage A backbone intact.
- But it gives back too much of the Stage B spread gains:
  - overall coverage drops materially
  - regime differentiation stays weak
  - pathwise jump realism gets worse again

So the live Stage B diagnosis is now sharper:
- scale-only adjustments are real and useful
- but even horizon-structured scale-only control is still too limited
- the hierarchy likely needs a distribution-shape mechanism beyond pure amplitude
  scaling if it is going to solve coverage/regime/jump behavior together

## Decision
- Keep the two-level hierarchy as the active framing.
- Close the scale-only Stage B subfamily as likely capped.
- Next step: post-experiment analysis comparing `277d`, `278a`, `279a`, `279b`,
  `280a`, and `280b` to decide the smallest non-scale Stage B mechanism that still
  preserves the fixed Stage A center path.
