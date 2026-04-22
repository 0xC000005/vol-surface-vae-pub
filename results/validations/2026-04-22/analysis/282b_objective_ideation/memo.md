# 282b Objective Ideation

## Context
`282a` proved that the Stage B failure was objective-related:
- weighted energy score prevents collapse and restores broad coverage
- but it is too global and over-disperses the scenario law relative to local fidelity

So the next objective should stay:
- exact
- proper
- defined on the same finite residual-support distribution

but it should become more local to the scalar coordinates.

## Decision
Next step: `282b-v0`

### Family
Same `281b` architecture, new objective only.

### Core idea
- keep the fixed `277d` Stage A center path
- keep the same fixed residual bank
- keep the same joint temperature+scale controller
- replace weighted energy score with exact weighted **CRPS** over the flattened scalar
  coordinates

For scalar support points `x_i` with weights `w_i`, optimize:
- `CRPS = sum_i w_i |x_i - y| - 0.5 sum_{i,j} w_i w_j |x_i - x_j|`

Then average across all path/cell coordinates.

### Why This Is The Smallest Principled Step
- no architectural change
- no new side loss
- same finite-support distribution
- proper score, but more local than energy score
- directly targets the `282a` failure mode

## Pre-Registered Success Criteria
Relative to `282a`, `282b` should:
- preserve broad coverage gains materially
- improve distributional fidelity, especially change-KS
- avoid catastrophic jump-scale inflation

## Immediate Next Action
Implement `282b-v0` by reusing the `281b` model class and replacing the training
objective with exact weighted CRPS.
