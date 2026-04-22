# 280a Hierarchical Ideation

## Context
The `278a` / `279a` / `279b` sequence now gives a clean Stage B diagnosis:

- `278a`: enough spread, not enough Stage A center preservation
- `279a`: enough Stage A center preservation, not enough spread/fidelity preservation
- `279b`: partial centering recovers some spread, but centering strength alone still
  trades off against mean-reversion and jump realism

So the next move should not be another new family. The next move should be the
smallest mechanism that adjusts **residual amplitude** without moving the fixed Stage A
center path.

## Decision
Next step: `280a-v0`

### Family
Center-preserving hierarchical retrieval with learned residual scaling.

### Core idea
- keep the fixed `277d` deterministic Stage A center path
- keep the same anchored retrieved residual bank
- keep partial residual centering as the Stage B residual geometry
- add a small query-conditioned scale head that only scales the residual scenarios
- do **not** let the learned component weight or shift whole future paths

Form:
- `scenarios = center + scale(query) * adjusted_residuals`

Possible minimal parameterizations:
- one scalar per query
- or one short horizon profile shared across cells

The default first attempt should be the smallest one:
- one scalar per query

### Why This Is The Smallest Principled Step
- the deterministic Stage A center path remains fixed
- the retrieval residual bank remains fixed
- the only learned freedom is residual amplitude
- this directly targets the current undercoverage / regime-width bottleneck without
  reintroducing center drift

## Pre-Registered Success Criteria
Relative to `279b`, `280a` should:
- preserve surface, cointegration, and cross-cell passes
- recover coverage and regime coverage materially
- avoid losing the Stage A center enough to destroy mean reversion again
- improve either conditionality or pathwise jump realism

## Immediate Next Action
Implement `280a-v0` with a single query-conditioned residual scale head and evaluate it
on the full 11-suite.
