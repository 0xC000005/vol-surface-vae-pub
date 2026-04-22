# 287b Retrieval Coordinate Ideation

## Context
`287a` closed one specific hypothesis:
- aligning retrieval and support in a pure local-history level coordinate is **not**
  enough
- the deterministic center path still loses too much current-state anchoring

The negative is narrow, not fatal. It does **not** say the local-history coordinate is
wrong. It says the support primitive inside that coordinate is wrong.

## Decision
Next step: `287b-v0`

### Family
Local-history-coordinate retrieval with normalized change replay.

### Core idea
Keep the same upstream reset as `287a`:
- retrieval trained in the local-history coordinate
- candidate bank stored in that coordinate

But change the support primitive:
- do **not** decode absolute future levels in z-space
- instead store normalized future changes `delta_z`
- reconstruct the future by replaying those normalized changes from the query's
  current normalized state

This restores the useful part of `277d`:
- current-state anchoring

while keeping the useful part of the `286` / `287a` line:
- local-history coordinate alignment

### Why This Is The Smallest Principled Follow-Up
- same retrieval backbone
- same two-level program
- same local-history coordinate
- no new scorer
- no new stochastic generator
- only changes the support primitive inside the already selected coordinate

### Immediate Next Action
Implement `287b-v0` by storing normalized future changes in the Stage A bank and
reconstructing deterministic futures via normalized change replay from the query's
current normalized state.
