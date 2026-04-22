# 277b Postmortem

## Result
- Full 11-suite score: 3/11
- Passes: surface, block_ar, cross_cell_correlation

## What Matters For Stage A
`277b` keeps the retrieval family alive.

The important question was whether we could preserve the strong dynamic-law realism of
`277a` while fixing its level-path copying pathology by re-anchoring the retrieved
future changes to the query window's current level.

## Key Metrics
- surface:
  - all surface gates pass
- time_series:
  - ACF corr: 0.867
  - kurtosis ratio: 0.974
  - q99(|ΔIV|) cells passing: 19/25
- cointegration:
  - ratio: 1.160
  - worst-cell ratio: 0.211
- distributional_fidelity:
  - level KS pass: 3/25
  - change KS pass: 21/25
  - median-bias pass: 25/25
  - MAE pass: 22/25
- cross_cell_correlation:
  - corr ratio: 1.177
  - rank ratio: 0.917
- mean_reversion:
  - aggregate ratio: 0.360
  - active cells: 3/24
  - active-cell corr: 0.286
- pathwise_jump_realism:
  - max-jump KS: 0.396
  - q90 ratio: 0.928
  - q99 ratio: 0.976

## Mechanism Read
- `277b` fixed the exact `277a` anchoring issue only partially.
- Re-anchoring the retrieved future **changes** to the query level preserved most of
  the dynamic-law benefits:
  - change KS stayed strong
  - ACF and kurtosis stayed in range
  - cross-cell structure stayed strong
- But the full hard re-anchor also removed too much of the retrieved path's own
  long-run anchor:
  - mean reversion collapsed from over-strong to too weak
  - level KS improved only marginally
  - jump-shape realism worsened
- So the live Stage A bottleneck is no longer “copying a real path is too literal.”
- It is now:
  - raw L2 retrieval in history space finds a path with the right change law,
  - but the retrieval metric is not predictive enough of the *right anchored future*
    for the query.

## Decision
- Keep the two-level reset.
- Keep retrieval as the active deterministic Stage A family.
- Do not add another manual anchoring knob.
- Next step: `277c-v0`, learned history embedding retrieval.
- Rationale:
  - the retrieval family is still the strongest deterministic Stage A line
  - the remaining miss is now the similarity metric, not the backbone
  - a learned retrieval embedding is the smallest first-principles extension that
    lets the model learn which histories imply the right anchored future changes
    without adding explicit low-rank, EC, or bounded side paths
