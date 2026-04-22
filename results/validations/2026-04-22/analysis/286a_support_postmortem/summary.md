# 286a Support Postmortem

## Result
- Full 11-suite score: `4/11`
- Passes: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`

## What Was Tested
`286a` kept the existing `283a` query-conditioned reweighting checkpoint fixed and
changed only the support object.

Instead of sampling:
- raw retrieved future paths, or
- deltas replayed from the query last level,

it transported each retrieved future through a local affine history coordinate:
- library future standardized by its own recent-history mean and level std
- then mapped into the query's recent-history mean and level std

## Mechanism Read
This is a real new regime, not another interpolation failure.

Relative to the raw-future support probe `284b`, `286a` recovers much more of the
statistical-validity object:
- cointegration stays a pass
- cross-cell correlation stays a pass
- full-horizon mean-reversion profile passes again
- surface validity remains clean

At the same time it keeps the main local-fidelity gain of the freer-support family:
- change KS remains extremely strong (`24/25`)
- coverage and calibration stay usable

But the core deterministic carryover miss remains:
- level KS is still `0/25`
- window-floor still fails
- jump realism still fails
- short-horizon active mean-reversion support still fails

So the new support coordinate is directionally right, but full affine
mean-and-scale transport is still not the central level law the hierarchy needs.

## Decision
- Keep the new history-coordinate support family alive.
- Do not go back to raw-vs-anchored interpolation.
- Next step should isolate whether the remaining miss comes from the **scale**
  part of the affine transport:
  - test a mean-only history transport before escalating to a more complex support
    family.
