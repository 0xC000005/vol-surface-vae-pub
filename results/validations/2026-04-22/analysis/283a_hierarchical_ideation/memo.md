# 283a Hierarchical Ideation

## Context
`282b` is now the best-behaved Stage B objective in the fixed-center residual family:
- it prevents the `281b` collapse
- it restores the `282a` coverage gains more cleanly
- it brings mean reversion back through gate

But the deeper cap is now explicit:
- every Stage B variant from `278a` through `282b` still has `0/25` level-KS passes
- the current fixed-center residual hierarchy cannot repair the deterministic support
  mismatch inherited from the `277d` anchor family

So the next step should not be another residual objective tweak.

## Decision
Next step: `283a-v0`

### Family
Reopen raw future-path weighting, but with a proper score instead of the old
target-matching KL objective.

### Core idea
- keep the same `277d` learned retrieval embeddings and top-k candidate future set
- drop the fixed-center residualization requirement
- let Stage B learn a query-conditioned distribution directly over retrieved future
  paths
- train that distribution with exact weighted CRPS over the finite future-path support
  instead of KL to a soft nearest-path target

### Why This Is The Smallest Principled Shift
- no new encoder
- no new decoder
- no handcrafted residual centering
- no new side losses
- only one substantive change:
  - the distribution is allowed to move the center when needed
  - and it is trained with the best local proper-score direction found in `282b`

### What This Tests
Whether the real cap was:
- the residual objective only, or
- the fixed-center residual decomposition itself

`283a` is the first clean test of the second hypothesis.

## Pre-Registered Success Criteria
Relative to `282b`, `283a` should improve at least two of:
- level KS pass cells
- change KS pass cells
- regime-sensitive width allocation
- pathwise jump realism

while keeping:
- surface validity
- cross-cell structure
- mean reversion within reasonable range

## Immediate Next Action
Implement `283a-v0` by reusing the `278c` reweighting architecture and replacing the
old target-matching KL objective with exact weighted CRPS over the retrieved
future-path support.
