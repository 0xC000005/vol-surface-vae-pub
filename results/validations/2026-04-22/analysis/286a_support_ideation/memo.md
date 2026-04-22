# 286a Support Ideation

## Context
The current Stage A support-family bracket is now explicit:

- anchored-delta support preserves statistical validity, especially mean reversion
  and cointegration, but leaves level KS effectively dead
- raw future support improves level fidelity and window-floor behavior, but breaks
  mean reversion badly
- simple interpolation between those two support objects does not reconcile the
  tradeoff

So the next step should not be another weighting tweak and not another blend inside
the same coordinate system. The support object itself should change.

## Decision
Next step: `286a-v0`

### Family
History-affine support transport.

### Core idea
Represent each retrieved future path in the affine coordinate of its own recent
history rather than in raw absolute levels or as deltas replayed from the last
level.

For each library window:
- compute recent-history mean `mu_lib`
- compute recent-history level std `sigma_lib`
- express the retrieved future as
  - `z_t = (future_t - mu_lib) / sigma_lib`

For the query history:
- compute `mu_q`
- compute `sigma_q`
- reconstruct the candidate support as
  - `future_t = mu_q + sigma_q * z_t`

This creates a genuinely new support object:
- not last-level anchoring
- not raw absolute future levels
- not a fixed interpolation between the two

### Why This Is The Smallest Principled Step
- no new scorer
- no new loss
- no new bank
- no new branch-heavy architecture
- no finance-specific side path

Only the support coordinate changes:
- from absolute future levels or replayed deltas
- to a local affine coordinate defined by recent history statistics

### Why This Fits The Mechanism Read
The current failure looks like a mismatch in what should be preserved across
retrieved futures:
- raw future support preserves too much library-specific absolute level law
- anchored-delta support preserves too much query-specific last-level anchoring

The missing middle is likely a support object that preserves:
- library future shape relative to its own recent state
- while adapting the absolute level law to the query's recent state

History-affine transport is the cleanest first-principles version of that idea.

### Pre-Registered Success Criteria
Relative to `284b` and `285a`, `286a` should improve at least one of:
- mean-reversion active support
- cointegration local robustness
- regime-sensitive width allocation

while preserving:
- surface validity
- change KS strength
- materially better level fidelity than the anchored-delta family

### Immediate Next Action
Implement `286a-v0` as an evaluation-only support-object probe using the existing
`283a` reweighting checkpoint and the new history-affine support transport.
