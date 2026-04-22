# 277e Postmortem

## Result
- Full 11-suite score: 5/11
- Passes: surface, block_ar, cointegration, cross_cell_correlation, mean_reversion

## What Was Tested
`277e` kept the `277d` learned retrieval family fixed and changed only the
deterministic selection rule:
- retrieve top-k plausible futures
- choose the medoid future in learned future-embedding space
- anchor that future to the query level

## Mechanism Read
- This was the right test for top-1 brittleness.
- The result is basically neutral-to-slightly worse versus `277d`:
  - mean reversion stayed a pass overall, but active support weakened
  - level KS improved only marginally
  - jump realism did not improve
- So the live deterministic bottleneck is **not** just that top-1 retrieval picks an
  overly idiosyncratic future.
- The remaining deterministic miss is deeper:
  - exact path selection alone will not fix level-distribution fidelity or tail-shape
    realism

## Decision
- Close `277e` as a negative local selection-rule experiment.
- Keep `277d` as the deterministic Stage A frontier.
- Move to the explicit two-level hierarchical scenario layer on top of `277d`.
