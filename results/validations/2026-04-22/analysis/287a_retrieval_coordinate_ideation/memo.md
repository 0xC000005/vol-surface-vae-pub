# 287a Retrieval Coordinate Ideation

## Context
The local support-transport bracket is now closed:

- `284b`: raw future support improves level fidelity but breaks mean reversion
- `285a`: simple transport between raw future and anchored-delta does not reconcile
  the tradeoff
- `286a`: history-affine support transport is a real new regime, but still leaves
  level KS dead
- `286b`: removing the scale term does not help

So the bottleneck is no longer a support transport choice on top of the existing
`277d` retrieval bank. The issue has moved upstream into the retrieval coordinate and
the candidate bank itself.

## Decision
Next step: `287a-v0`

### Family
Local-history-coordinate retrieval and support bank.

### Core idea
Train the Stage A retrieval backbone and construct the candidate bank directly in the
same local-history coordinate that made the `286a` and `286b` probes interpretable.

For each history/future window:
- compute recent-history statistics from the history window
- represent the future in that local coordinate
- train retrieval similarity on that coordinate
- store candidate futures in the same coordinate

At inference:
- retrieve top-k candidates in the local-history coordinate
- sample or select support in that same coordinate
- map back to the query state only once, at the end

This removes the current mismatch:
- retrieval was trained in one future representation
- support was later transported post hoc in another

### Why This Is The Smallest Principled Upstream Change
- same two-level hierarchy
- same nonparametric retrieval idea
- no new stochastic neural generator
- no new side branches
- no explicit finance-specific structure

The only substantive change is to align:
- retrieval training coordinate
- candidate bank coordinate
- support generation coordinate

### Why This Fits The Evidence
The `286a` / `286b` bracket says the history-coordinate idea is not wrong, but
post-hoc transport on top of the old `277d` bank is too weak.

So the next live hypothesis is:
- the right support family may only appear once the retrieved candidate set itself is
  organized in the same coordinate

### Immediate Next Action
Implement `287a-v0` as a new Stage A retrieval backbone trained on a local-history
future representation, then evaluate the deterministic center path before reopening
Stage B weighting.
