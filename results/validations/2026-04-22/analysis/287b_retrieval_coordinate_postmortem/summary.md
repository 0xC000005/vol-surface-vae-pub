# 287b Retrieval Coordinate Postmortem

## Result
- Full 11-suite score: `3/11`
- Passes: `surface`, `block_ar`, `cross_cell_correlation`

## What Was Tested
`287b` kept the upstream `287a` reset:
- retrieval trained in the local-history coordinate
- candidate bank stored in that same coordinate

But it changed the support primitive:
- instead of decoding absolute future levels in the local-history coordinate
- it stored normalized future changes and replayed them from the query's current
  normalized state

This was the cleanest way to restore current-state anchoring without going back to
the old raw-level bank.

## Mechanism Read
This fixed the specific `287a` anchoring failure, but it overshot toward a different
tradeoff.

What improved versus `287a`:
- retrieval validation improved materially during training
- mean-reversion aggregate profile recovered into gate
- change KS became perfect (`25/25`)
- level KS improved further (`3/25`)
- jump realism improved sharply (`max-jump KS = 0.229`)
- support validity remained clean

What regressed:
- local cointegration robustness fell back out of gate
- active-cell mean-reversion support still failed
- deterministic coverage is still zero, as expected for a deterministic Stage A path

So normalized change replay is directionally right:
- the local-history coordinate itself is not the problem
- the absolute-level support primitive in `287a` was the wrong one

But the new family is still missing a **slow structural anchor**:
- local change law is much better
- central level law is better
- but long-run joint structure at the hard cells regresses

## Decision
- Keep the upstream local-history retrieval reset alive.
- Keep normalized change replay as the right support primitive inside that family.
- Next step should add the smallest missing slow-structure component inside the same
  coordinate, rather than abandoning the family:
  - a representation that combines normalized change replay with a slow cumulative
    displacement anchor in the local-history coordinate.
