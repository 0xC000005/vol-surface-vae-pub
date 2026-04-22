# 277d Postmortem

## Result
- Full 11-suite score: 5/11
- Passes: surface, block_ar, cointegration, cross_cell_correlation, mean_reversion

## Why This Matters
`277d` is the first clean deterministic Stage A breakthrough in the strict two-level
reset.

It beats the archived `4/11` frontier and, more importantly, validates the Stage A
retrieval program:
- retrieval is the right deterministic family
- learned similarity is useful
- the future key must carry both local changes and anchored level displacement

## Key Metrics
- surface:
  - all surface gates pass
- time_series:
  - ACF corr: 0.928
  - kurtosis ratio: 0.913
  - q99(|ΔIV|) cells passing: 16/25
- cointegration:
  - ratio: 0.766
  - worst-cell ratio: 0.281
- distributional_fidelity:
  - level KS pass: 0/25
  - change KS pass: 23/25
  - median-bias pass: 25/25
  - MAE pass: 22/25
- cross_cell_correlation:
  - corr ratio: 1.058
  - rank ratio: 1.027
- mean_reversion:
  - aggregate ratio: 1.065
  - active cells: 21/24
  - active-cell corr: 0.750
- pathwise_jump_realism:
  - max-jump KS: 0.391
  - q90 ratio: 0.792
  - q99 ratio: 0.928

## Mechanism Read
- `277d` fixed the core `277c` pathology.
- Once the learned retrieval key saw both:
  - future daily changes
  - cumulative displacement from the current state

the deterministic Stage A path regained:
- cointegration
- strong mean reversion
- stable cross-cell structure
- strong change-law realism

So the two-level program is now much better justified:
- Stage A can in fact learn the deterministic statistical validity object
- the remaining six misses are much cleaner

## Remaining Deterministic Bottleneck
The deterministic misses are now concentrated in:
- level KS distribution fit
- tail-shape / extreme-jump realism
- a few time-series tail cells

This means the current top-1 retrieved future is still too brittle:
- it preserves a realistic path family
- but a single retrieved path is not yet the right deterministic **central** path for
  level distribution fidelity

## Decision
- Keep the two-level reset.
- Keep learned retrieval as the active deterministic Stage A family.
- Next step: `277e-v0`, top-k learned retrieval with future-embedding medoid
  selection.

## Why 277e Is The Smallest Principled Next Step
- no averaging of paths
- no low-rank side head
- no bounded anchor blend
- no explicit finance-specific structure

It only changes the deterministic selection rule:
- from top-1 nearest neighbor
- to central-path medoid selection within the top-k plausible futures

That is the smallest way to reduce deterministic brittleness without reintroducing
smooth learned predictors.
