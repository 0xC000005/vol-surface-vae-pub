# 283a Postmortem

## Result
- Full 11-suite score: `3/11`
- Passes: `surface`, `block_ar`, `cross_cell_correlation`

## What Was Tested
`283a` reopened the old `278c` raw future-path weighting family, but replaced the
old target-matching KL objective with exact weighted CRPS.

## First Read
- Relative to `282b`, `283a` improved several local distributional properties:
  - `change KS` jumped back to `23/25`
  - `level KS` moved from `0/25` to `1/25`
  - kurtosis moved into gate
  - regime differentiation passed again
- But it also gave back too much of the Stage A statistical-validity object:
  - cointegration local worst-cell gate failed
  - mean-reversion active support collapsed
  - jump realism worsened

## Critical Mechanism Finding
The experiment is not a clean verdict on the reopened family, because the training
objective and inference path are mismatched.

- Training optimized the weighting law over the **raw retrieved future paths**
- Inference still samples the **anchored future paths** produced by applying retrieved
  deltas to the query last level

So `283a` was not actually training the same support distribution it later sampled at
evaluation time.

## Decision
- Do not interpret `283a` as a decisive negative on raw future-path reweighting.
- Next step: `283b-v0`, same architecture and same weighted CRPS objective, but train
  on the **anchored candidate futures** that match the inference geometry exactly.
