# 277c Stage A Ideation

## Context
`277a` and `277b` established that deterministic retrieval is the first genuinely live
Stage A family in the strict two-level reset:
- retrieval preserves real dynamic law much better than learned deterministic
  predictors
- the remaining deterministic miss is no longer the path family itself
- it is the retrieval **metric**

`277b` also showed that adding another manual anchoring rule is not the right next
move. Hard query-level re-anchoring fixed the old level-copy pathology only partially
and overcorrected mean reversion.

## Decision
Next step: `277c-v0`

### Family
Learned history-embedding retrieval for deterministic future-change paths.

### Core idea
- keep the deterministic retrieval backbone
- keep query-level anchoring of retrieved future **changes**
- replace raw L2 distance on normalized history tensors with a learned embedding
  distance

## Proposed Model
Two small encoders:
- `f(H)` history encoder
- `g(ΔF)` future-change encoder

Train them with a contrastive objective so that a history embedding is close to the
embedding of its own future-change path and far from mismatched futures.

At inference:
- encode query history with `f(H_q)`
- encode all training histories once and store their embeddings
- retrieve nearest neighbors in learned embedding space
- use the retrieved future-change path and anchor it to the query's last observed
  level, as in `277b`

## Why This Is The Smallest Principled Step
- no hard low-rank head
- no bounded idio path
- no EC baseline
- no manual blend coefficient
- no explicit finance-specific assumptions

This is still first-principles:
- a memory-based deterministic Stage A model
- but the similarity function is learned from data rather than hand-fixed

## Minimal Training Story
- training data: `(history, future_change_path)`
- build `future_change_path` as raw daily level differences in `[0,1]` space
- encoder output dimension: small latent embedding, e.g. `d=64`
- objective: symmetric InfoNCE over batch
- no reconstruction decoder
- no KL / prior / posterior

## Expected Failure Mode To Test
If `277c` still fails badly on level KS and MR while preserving change KS and
cross-cell structure, then the retrieval metric is not the main issue and Stage A may
need a different deterministic scaffold than direct analog retrieval.

## Pre-Registered Success Criteria
Relative to `277b`, `277c` should improve at least two of:
- level KS pass cells
- worst-cell cointegration ratio
- aggregate MR ratio
- active MR pass count

while preserving:
- change KS pass cells >= 18
- corr ratio inside gate
- rank ratio inside gate
- surface validity pass

## Immediate Next Action
Implement `277c-v0` in fresh files and run the full deterministic Stage A iteration.
