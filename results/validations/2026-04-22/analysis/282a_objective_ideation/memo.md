# 282a Objective Ideation

## Context
`281b` closed the current Stage B **expected-distance** objective.

What is now established:
- the fixed `277d` Stage A center path is still the right anchor
- the residual bank is expressive enough to support useful spread
- the minimal joint temperature+scale Stage B controller has enough freedom
- but the expected-distance objective collapses the residual law because it rewards one
  narrow good-match scenario rather than a usable scenario distribution

So the next step should not change the hierarchy or the residual bank.
It should change only the Stage B training objective.

## Decision
Next step: `282a-v0`

### Family
Same `281b` architecture, new objective.

### Core idea
- keep the fixed `277d` Stage A center path
- keep the same top-k residual bank
- keep the same minimal joint temperature+scale controller
- replace expected residual distance with an exact **weighted energy score** over the
  finite residual-support distribution

For a discrete weighted ensemble `x_i` with weights `w_i`, optimize:
- `ES = sum_i w_i ||x_i - y|| - 0.5 sum_{i,j} w_i w_j ||x_i - x_j||`

This is the smallest principled correction because:
- it is a proper scoring rule
- it directly penalizes distribution collapse
- it does not require new architecture or extra side losses

## Pre-Registered Success Criteria
Relative to `281b`, `282a` should:
- avoid catastrophic coverage collapse
- preserve the Stage A center-path suites
- improve either coverage or distributional fidelity materially
- keep regime differentiation at least close to `281a`

## Immediate Next Action
Implement `282a-v0` by reusing the `281b` model class and training it with the exact
weighted energy score instead of expected-distance matching.
