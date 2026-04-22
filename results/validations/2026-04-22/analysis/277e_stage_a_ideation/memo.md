# 277e Stage A Ideation

## Context
`277d` is the first deterministic Stage A breakthrough:
- 5/11 overall
- first reset-line model to beat the archived 4/11 frontier
- passes cointegration, cross-cell structure, and mean reversion together

The remaining deterministic failures are now narrower:
- level KS is still dead
- jump-shape realism is still weak
- a few time-series tail cells remain outside gate

The current read is that the learned retrieval family is right, but **top-1** path
selection is too brittle to serve as the deterministic central path.

## Decision
Next step: `277e-v0`

### Family
Top-k learned retrieval with medoid selection in learned future-embedding space.

### Core idea
- keep the `277d` learned retrieval model unchanged
- for each query history, retrieve top-k plausible future candidates
- choose the **medoid** candidate among those futures in learned embedding space
- anchor that medoid future path to the query level exactly as before

## Why This Is The Smallest Principled Step
- no path averaging
- no decoder
- no side path
- no explicit finance assumptions

Only the deterministic selection rule changes:
- top-1 nearest future can be too brittle
- medoid selection keeps a real observed path while choosing a more central future
  among plausible candidates

## Pre-Registered Success Criteria
Relative to `277d`, `277e` should improve at least two of:
- level KS pass cells
- max-jump KS
- tail-scale pass cells
- worst per-cell MAE

while preserving:
- surface validity pass
- corr ratio inside gate
- rank ratio inside gate
- cointegration pass
- mean reversion pass

## Immediate Next Action
Implement `277e-v0` and run the full deterministic Stage A iteration.
