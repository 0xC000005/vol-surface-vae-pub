# 282a Hierarchical Postmortem

## Result
- Full 11-suite score: 4/11
- Passes: surface, block_ar, cointegration, cross_cell_correlation

## What Was Tested
`282a` kept the exact `281b` architecture and residual bank, but replaced the
expected-distance objective with an exact **weighted energy score** over the finite
scenario-support distribution.

## Mechanism Read
- This is a clean objective effect.
- Relative to `281b`, the weighted energy score does exactly what it should:
  - prevents collapse
  - restores broad coverage strongly
  - restores all-horizon coverage pass
  - passes the severe undercoverage layer in regime coverage
- But it over-corrects:
  - change-KS fidelity collapses badly
  - tail scale becomes too large
  - jump realism worsens again
  - full-horizon mean_reversion falls just below the gate

So the objective diagnosis is now sharper:
- expected-distance is too collapse-seeking
- energy score is more calibration-seeking, but too global for local fidelity

The next objective should stay proper and exact on the same finite support, but become
more local to the marginal/path coordinates.

## Decision
- Keep the same two-level hierarchy and the same fixed residual-support distribution.
- Close weighted energy score as too global by itself.
- Next step: `282b-v0`, exact weighted **CRPS** over the same residual-support
  distribution.
