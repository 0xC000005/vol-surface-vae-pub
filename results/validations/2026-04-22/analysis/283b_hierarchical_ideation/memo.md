# 283b Hierarchical Ideation

## Context
`283a` reopened the raw future-path weighting family with weighted CRPS, but the run
was not yet a clean test of that family.

The key issue is a train/inference mismatch:
- training optimized the weighting law over raw retrieved future paths
- inference samples anchored future paths obtained by replaying retrieved deltas from
  the query last level

So the next step should correct that mismatch before making any family-level decision.

## Decision
Next step: `283b-v0`

### Family
Same `278c/283a` reweighting architecture, same weighted CRPS objective, corrected
support geometry.

### Core idea
- keep the same `277d` retrieval embeddings and top-k candidate set
- keep the same reweighting scorer
- keep the same exact weighted CRPS objective
- change only the support used during training:
  - train on the **anchored candidate futures** that exactly match the inference-time
    sampling geometry

### Why This Is The Smallest Principled Fix
- no new architecture
- no new loss
- no new bank
- no new side path
- only removes the now-identified train/inference mismatch

## Immediate Next Action
Implement `283b-v0` by copying `283a` and replacing the raw candidate support with the
anchored candidate support used by `sample_batched`.
