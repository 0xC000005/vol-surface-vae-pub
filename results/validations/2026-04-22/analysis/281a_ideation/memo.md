# 281a Hierarchical Ideation

## Context
The Stage B comparison is now decisive:

- raw retrieved futures (`278a`) have the best spread/fidelity signal
- center-preserving residualization (`279a`) restores the deterministic backbone
- centering-strength and scale-only controls (`279b`, `280a`, `280b`) are all real,
  but they are still too blunt

So the next mechanism should not rescale the residual bank again.
It should change the **selection shape** over that bank while keeping the fixed Stage A
center path intact.

## Decision
Next step: `281a-v0`

### Family
Center-preserving hierarchical retrieval with learned residual temperature.

### Core idea
- keep the fixed `277d` deterministic Stage A center path
- keep the same anchored top-k residual bank
- keep the same Stage A retrieval scores as the base logits
- learn only one query-conditioned scalar: residual sampling temperature
- convert the logits into weights
- center the residual bank under those same weights
- sample residual scenarios from that weighted-centered bank

Formally:
- `w(query) = softmax(score / tau(query))`
- `resid_centered = resid - sum_k w_k resid_k`
- `scenario = center + resid_centered[sampled_k]`

### Why This Is The Smallest Principled Step
- no new scorer
- no new side path
- no new explicit factor assumptions
- no scale-only distortion of the residual geometry
- center preservation is enforced by centering under the same sampling distribution

## Pre-Registered Success Criteria
Relative to `280b`, `281a` should:
- preserve surface, cointegration, cross-cell structure, and mean_reversion
- improve either coverage or regime coverage materially
- improve pathwise jump realism relative to `280b`
- avoid the raw center drift failure from `278a`

## Immediate Next Action
Implement `281a-v0` with a learned query-conditioned temperature head and train it by
minimizing expected residual distance to the realized future under the weighted-centered
residual bank.
