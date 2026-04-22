# 277c Postmortem

## Result
- Full 11-suite score: 3/11
- Passes: surface, block_ar, cross_cell_correlation

## What Matters For Stage A
`277c` tested the cleanest next deterministic retrieval step after `277b`:
- keep retrieval
- keep anchored future changes
- replace raw L2 history distance with a learned history-to-future similarity

This was the right test because `277b` showed that the active deterministic bottleneck
had moved from the path family itself to the retrieval metric.

## Key Metrics
- surface:
  - all surface gates pass
- time_series:
  - ACF corr: 0.923
  - kurtosis ratio: 0.887
  - q99(|ΔIV|) cells passing: 19/25
- cointegration:
  - ratio: 0.635
  - worst-cell ratio: 0.077
- distributional_fidelity:
  - level KS pass: 0/25
  - change KS pass: 19/25
  - median-bias pass: 17/25
  - MAE pass: 20/25
- cross_cell_correlation:
  - corr ratio: 0.958
  - rank ratio: 1.201
- mean_reversion:
  - aggregate ratio: 0.458
  - active cells: 1/24
  - active-cell corr: -0.233
- pathwise_jump_realism:
  - max-jump KS: 0.589
  - q90 ratio: 0.882
  - q99 ratio: 0.882

## Mechanism Read
- `277c` confirms that retrieval remains the right deterministic Stage A family:
  - surface validity stayed strong
  - change-law realism stayed usable
  - cross-cell structure stayed inside gate
- But the learned metric was trained only against the **future change path**.
- That made the retrieval key too level-free:
  - level KS collapsed from weak to dead
  - median-bias and cointegration worsened
  - active mean reversion also stayed dead
- So the clean conclusion is not “learned retrieval is wrong.”
- The clean conclusion is:
  - change-only future targets are too incomplete for deterministic Stage A retrieval
  - the retrieval key must encode both local change law and long-run anchored level

## Decision
- Keep the two-level reset.
- Keep retrieval as the active deterministic Stage A family.
- Keep learned similarity as the next live idea.
- Next step: `277d-v0`, learned retrieval with a richer future target that includes both:
  - future daily changes
  - future cumulative level displacement from the current state

## Why 277d Is The Smallest Principled Fix
- still no hard low-rank head
- still no bounded side paths
- still no EC baseline
- still no explicit finance-specific assumptions
- only the retrieval target representation changes, because `277c` showed the change-only representation is the actual bottleneck
