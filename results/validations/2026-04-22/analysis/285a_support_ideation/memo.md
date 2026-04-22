# 285a Support Ideation

## Context
The current support-family split is now explicit:
- raw future support helps level-side fidelity (`284b`)
- anchored-delta support preserves more statistical validity (`282b` / `283b`)
- the local weighting and centering bracket does not reconcile those two objects

So the next step should change the **support object itself**, not the weighting law.

## Decision
Next step: `285a-v0`

### Family
Deterministic support-transport interpolation between raw future support and
anchored-delta support.

### Core idea
For each retrieved future path:
- let `offset = query_last_level - library_last_level`
- construct the candidate support as:
  - `future_t = raw_future_t + beta_t * offset`
- where `beta_t` decays from `1` at the first future step to `0` at the last horizon

This creates a new support object:
- early horizon stays aligned to the current state
- long horizon reverts toward the raw retrieved future level law

### Why This Is The Smallest Principled Step
- no new training
- no new scorer
- no new loss
- no new bank
- only one new assumption:
  - the relevance of the current level offset should decay over horizon

It is the minimal support object that sits between the two already-tested extremes.

## Immediate Next Action
Implement `285a-v0` as an evaluation-only support-object variant using the existing
`283a` reweighting checkpoint and the new horizon-decayed offset transport support.
