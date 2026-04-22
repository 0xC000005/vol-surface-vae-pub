# 283b Postmortem

## Result
- Full 11-suite score: `4/11`
- Passes: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`

## What Was Tested
`283b` kept the `283a` architecture and weighted CRPS objective, but corrected the
support geometry so that training used the same anchored candidate futures that
inference later samples.

## Mechanism Read
- The train/inference mismatch fix was real:
  - cointegration local worst-cell support recovered to pass
  - the result is cleaner than `283a`
- But the family still sits between the current two endpoints:
  - it keeps `283a`'s strong change-KS behavior
  - but it still gives back too much mean-reversion active support and jump realism
  - level-KS improves only marginally (`1/25`)

So `283b` is the first clean verdict on the reopened weighting family:
- letting the support distribution move the center helps local fidelity
- but letting it move fully is still too much for the Stage A statistical-validity
  object

## Decision
- Keep the reopened weighting family alive.
- Next step: test the smallest explicit interpolation between the current endpoints.
- `283c-v0`: same learned weighting law and same weighted CRPS objective, but apply a
  fixed partial centering factor to the anchored candidates around the `277d`
  deterministic center path.
