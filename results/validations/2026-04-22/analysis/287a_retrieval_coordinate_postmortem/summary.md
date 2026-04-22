# 287a Retrieval Coordinate Postmortem

## Result
- Full 11-suite score: `4/11`
- Passes: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`

## What Was Tested
`287a` was the first upstream reset after closing the `286a` / `286b` support bracket.

It changed the Stage A family itself:
- retrieval training moved into a local-history coordinate
- the candidate bank was stored in that same coordinate
- deterministic support generation mapped candidates back into the query state only
  once, at the end

So unlike `286a` / `286b`, this was not a post-hoc transport on top of the old
`277d` bank. Retrieval and support were finally aligned in the same coordinate.

## Mechanism Read
This is a clean negative on the **pure local-history level coordinate**.

What improved:
- the model remains support-clean
- cross-cell structure still passes
- cointegration still passes
- level KS improves slightly versus the old deterministic center path (`1/25` instead
  of `0/25`), but remains far from usable

What broke:
- deterministic coverage is exactly zero
- mean-reversion degrades badly
- active-cell slope support collapses
- jump realism degrades

The likely cause is specific:
- the local-history coordinate removed too much of the current-state anchoring
- retrieval and support now align, but they align around an object that is too
  insensitive to the query's current normalized state

So the issue is not that the local-history idea is useless.
It is that **absolute future levels in that coordinate are still the wrong support
primitive**.

## Decision
- Close `287a` as a negative on pure local-history level retrieval.
- Keep the upstream reset active.
- Next step should preserve the local-history coordinate but restore current-state
  anchoring inside that coordinate:
  - replay normalized future changes from the query's current normalized state,
    rather than decoding absolute normalized future levels directly.
