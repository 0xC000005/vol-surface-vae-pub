# 285a Support Postmortem

## Result
- Full 11-suite score: `3/11`
- Passes: `surface`, `block_ar`, `cross_cell_correlation`

## What Was Tested
`285a` used the same `283a` reweighting checkpoint, but changed the support object to
a horizon-decayed offset transport:
- `future_t = raw_future_t + beta_t * (query_last - library_last)`
- `beta_t` decays linearly from `1` to `0`

## Mechanism Read
- The support-object interpolation effect is real:
  - `change KS` stays very strong (`24/25`)
  - level fidelity remains better than the anchored-delta family (`3/25`)
  - per-window coverage floor still passes
- But the hoped-for reconciliation does not happen:
  - cointegration local robustness regresses again
  - mean-reversion active support still fails badly
  - regime-sensitive width allocation stays weak

So the simple horizon-decayed offset transport is not enough to reconcile the two
support extremes.

## Decision
- Close the simple offset-decay support probe as a negative.
- The next principled step is no longer another simple transport tweak.
- Next step should be research ideation for a genuinely new Stage A support object or
  retrieval coordinate, not another local interpolation inside the current family.
