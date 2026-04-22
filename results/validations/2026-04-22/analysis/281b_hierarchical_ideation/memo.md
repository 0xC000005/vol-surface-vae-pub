# 281b Hierarchical Ideation

## Context
`281a` proves that residual temperature is a real Stage B mechanism class, but it also
shows the current split clearly:

- scale-only control helps broad amplitude-sensitive metrics
- temperature-only control helps regime differentiation and jump shape
- neither one alone solves Stage B

So the next step should not be a new family. It should be the smallest joint model that
lets Stage B control both amplitude and shape while still preserving the fixed Stage A
center path.

## Decision
Next step: `281b-v0`

### Family
Center-preserving hierarchical retrieval with learned residual temperature **and** one
learned residual scale.

### Core idea
- keep the fixed `277d` deterministic Stage A center path
- keep the same anchored top-k residual bank
- use a query-conditioned temperature over the fixed retrieval scores
- center the residual bank under those same weights
- apply one query-conditioned scalar scale to the centered residuals

Formally:
- `w(query) = softmax(score / tau(query))`
- `resid_centered = resid - sum_k w_k resid_k`
- `scenario = center + s(query) * resid_centered[sampled_k]`

### Why This Is The Smallest Principled Step
- no new scorer
- no new latent model
- no new side path
- same residual bank, same center-preservation rule
- only two scalar controls:
  - one for shape
  - one for amplitude

## Pre-Registered Success Criteria
Relative to `281a`, `281b` should:
- preserve regime differentiation pass
- preserve mean_reversion pass
- materially improve coverage or distributional fidelity
- avoid degrading jump realism materially

## Immediate Next Action
Implement `281b-v0` and evaluate whether the minimal joint amplitude+shape Stage B
control can move the hierarchy beyond the current `5/11` plateau.
