# 280b Hierarchical Ideation

## Context
`280a` validated learned residual scaling as a mechanism class, but also isolated the
next bottleneck:

- one scalar residual scale improves global amplitude-sensitive metrics
- but it is too blunt to allocate width correctly across regimes and horizons
- the fixed Stage A center path should still be preserved

So the next step should stay in the same hierarchy and only increase the scaling model
by one structural notch.

## Decision
Next step: `280b-v0`

### Family
Center-preserving hierarchical retrieval with a minimal query-conditioned horizon-scale
profile.

### Core idea
- keep the fixed `277d` deterministic Stage A center path
- keep the same anchored residual bank
- keep partial residual centering
- replace the single scalar scale head with a tiny query-conditioned scale profile over
  horizon

The cleanest first version is:
- predict 4 anchor scales for horizons `[1, 7, 14, 30]`
- linearly interpolate them across the 30-day horizon
- apply the resulting scale profile uniformly across cells

### Why This Is The Smallest Principled Step
- the Stage A center path remains fixed
- the residual bank remains fixed
- the learned freedom is still only residual amplitude
- horizon structure is added only because `280a` proved one scalar is too blunt

## Pre-Registered Success Criteria
Relative to `280a`, `280b` should:
- preserve surface, cointegration, and cross-cell passes
- preserve the coverage improvements
- recover regime differentiation materially above `1.10`
- improve either full-horizon mean_reversion or pathwise jump realism

## Immediate Next Action
Implement `280b-v0` with a 4-knot horizon-scale head and evaluate it on the full
11-suite.
