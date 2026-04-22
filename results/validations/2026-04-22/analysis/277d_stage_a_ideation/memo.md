# 277d Stage A Ideation

## Context
`277c` showed that learned retrieval is directionally valid, but its target was too
incomplete:
- using only the future change path made the learned similarity ignore the long-run
  level anchor
- deterministic Stage A then kept dynamic realism but lost level KS, bias, and active
  mean reversion

So the next step should not abandon retrieval or learned similarity.
It should fix the future target representation.

## Decision
Next step: `277d-v0`

### Family
Learned retrieval with a richer future target:
- future daily changes
- future cumulative displacement from the current state

### Retrieval action
Unchanged from `277b` / `277c`:
- retrieve a real future path from memory
- apply the retrieved daily changes on top of the query's current last observed level

## Proposed Model
Keep the same architecture:
- history encoder `f(H)`
- future encoder `g(R)`
- contrastive training

Change only the future representation `R`:
- `Δx_t`
- `x_t - x_0`

Concatenate them per future step and let the future encoder learn from that joint path
representation.

## Why This Is The Smallest Principled Step
- no extra decoder
- no manual anchor blend
- no low-rank head
- no bounded side path
- no explicit finance-specific assumptions

Only the target representation changes, because `277c` showed the representation
rather than the retrieval family is the active bottleneck.

## Pre-Registered Success Criteria
Relative to `277c`, `277d` should improve at least two of:
- level KS pass cells
- median-bias pass cells
- worst-cell cointegration ratio
- aggregate MR ratio

while preserving:
- change KS pass cells >= 18
- corr ratio inside gate
- rank ratio inside gate
- surface validity pass

## Immediate Next Action
Implement `277d-v0` and run the full deterministic Stage A iteration.
