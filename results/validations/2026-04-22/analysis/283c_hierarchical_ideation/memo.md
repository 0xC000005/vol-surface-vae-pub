# 283c Hierarchical Ideation

## Context
The `282b` vs `283b` bracket is now clean:
- `282b` preserves the `277d` center path too strongly and cannot fix the level-side
  distribution miss
- `283b` lets the center move enough to improve local fidelity, but it gives back too
  much mean-reversion active support and jump realism

So the next step should be the smallest explicit interpolation between those two
endpoints.

## Decision
Next step: `283c-v0`

### Family
Same reweighting architecture and same weighted CRPS objective, with fixed partial
centering of the anchored candidate futures.

### Core idea
- keep the same `277d` retrieval embeddings and top-k candidate set
- keep the same query-conditioned weighting scorer
- keep the same anchored candidate futures and exact weighted CRPS objective
- transform each anchored candidate future by:
  - `candidate' = center + alpha * (candidate - center)`
  - with fixed `alpha = 0.5`
  - where `center` is the deterministic `277d` center path

### Why This Is The Smallest Principled Step
- no new trainable head
- no new loss
- no new bank
- no new side path
- only tests whether the current tradeoff is primarily a support-geometry interpolation
  issue

## Immediate Next Action
Implement `283c-v0` by copying `283b` and replacing the anchored support with the
fixed half-centered anchored support.
