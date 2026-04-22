# 282b Postmortem

## Result
- Full 11-suite score: `5/11`
- Passes: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`, `mean_reversion`

## What Was Tested
`282b` kept the exact `281b` Stage B architecture and the same fixed `277d` center
path plus residual bank, but replaced the weighted energy score from `282a` with
exact weighted CRPS over the finite residual-support distribution.

## Mechanism Read
- This is a real improvement over `282a`.
- Weighted CRPS is local enough to recover the useful parts that weighted energy
  score had washed out:
  - score returns from `4/11` to `5/11`
  - full-horizon mean reversion passes again
  - broad coverage remains materially improved over the old collapse regime
  - calibration stays reasonable instead of exploding
- But the deeper cap is now explicit:
  - `level KS = 0/25` again
  - `change KS` remains weak at `5/25`
  - `max-jump KS` still fails at `0.378`
  - regime-sensitive width allocation remains too weak (`turb/calm = 1.071`)

## Cross-Variant Read
Across `278a` through `282b`, every Stage B variant still has `0/25` level-KS pass
cells. The objective changes are real, but they are not touching the deterministic
support problem.

That means the current fixed-center residual hierarchy is now structurally capped:
- the objective can trade off coverage, jump realism, and mean reversion
- but it cannot repair the level-distribution miss inherited from the support/anchor
  family

## Decision
- Close the fixed-center residual-objective branch as locally optimized enough.
- Do not run another Stage B residual objective tweak.
- Next step: reopen the raw future-path weighting family, but train it with a proper
  score instead of the old target-matching KL objective.
