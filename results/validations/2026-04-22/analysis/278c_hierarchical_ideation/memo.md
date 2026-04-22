# 278c Hierarchical Ideation

## Context
`278a` validated the two-level retrieval hierarchy.
`278b` showed that hand-built adaptive temperature from retrieval ambiguity is not
enough to fix the remaining Stage B allocation problem.

The remaining Stage B task is now clear:
- keep the same retrieved future set
- learn better weights over that set for each query

## Decision
Next step: `278c-v0`

### Family
Learned query-conditioned reweighting over the retrieved top-k future candidates.

### Core idea
- keep the `277d` retrieval embeddings fixed
- keep the same top-k future candidate set as `278a`
- train a small scoring module to reweight those candidates using:
  - query history embedding
  - candidate future embedding
  - their similarity

This stays within the same hierarchical retrieval program:
- no separate stochastic generator
- no explicit finance-specific regime features
- only a learned selection distribution over already retrieved plausible futures

## Immediate Next Action
Implement `278c-v0` and test whether learned candidate reweighting can improve
conditionality and regime coverage without breaking the Stage A center-path gains.
