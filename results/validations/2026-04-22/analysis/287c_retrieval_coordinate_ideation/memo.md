# 287c Retrieval Coordinate Ideation

## Context
`287b` fixed the exact `287a` anchoring failure:
- local-history coordinate is still viable
- normalized change replay is the right support primitive inside that family

But `287b` also exposed the next clean tradeoff:
- local change law, level KS, and jump realism all improve materially
- local cointegration robustness and active-cell mean-reversion support still fail

So the live bottleneck is no longer the support primitive.
It is the **retrieval key itself**:
- the current single embedding appears to overweight fast local change alignment
- and underpreserve the slow structural information needed for the hard cells

## Decision
Next step: `287c-v0`

### Family
Two-timescale local-history retrieval.

### Core idea
Keep the `287b` support primitive fixed:
- store normalized future changes
- replay them from the query's current normalized state

Change only the retrieval representation:
- encode fast normalized changes and slow cumulative displacement in separate channels
- combine them only at similarity time

This is the smallest way to test the current mechanism read:
- the local-history coordinate is right
- the normalized-change support primitive is right
- but a single compressed future key is washing out the slow structure

### Why This Is The Smallest Principled Step
- same two-level program
- same deterministic Stage A retrieval family
- same local-history coordinate
- same support primitive as `287b`
- no new stochastic model
- no ad hoc support transport

Only the retrieval key changes:
- from one mixed fast/slow embedding
- to a factored fast/slow embedding

### Immediate Next Action
Implement `287c-v0` as a dual-channel retrieval backbone in the same local-history
delta family, evaluate the deterministic center path, and compare directly against
`277d`, `287a`, and `287b`.
