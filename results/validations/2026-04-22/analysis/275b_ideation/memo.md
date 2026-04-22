# 275b Ideation

## Why 275a Is Not Enough
- `275a` is a good minimal deterministic baseline for support and static structure.
- It is too smooth and too weak dynamically:
  - zero change-KS passes
  - near-zero jump scale
  - weak active mean reversion
- That means the next Stage A change should target deterministic dynamic richness, not scenario width or support.

## Next Family
`275b-v0`: deterministic latent-state backbone with explicit observation model

## Core Model
- observation encoder: surface -> latent state / tokens
- deterministic latent transition: `z_t -> z_{t+1}`
- observation decoder: latent state -> next surface
- autoregressive rollout in latent space, deterministic only

## Why This Is The Smallest Principled Stage-A Extension
- It reuses the strongest Stage-A lesson from `272/273`: explicit observation modeling improved manifold fidelity materially.
- It avoids the stochastic latent-prior machinery that made the `272-274` lines unsuitable as a pure Stage-A deterministic backbone.
- It keeps the model simple:
  - no diffusion
  - no flow matching
  - no prior/posterior
  - no low-rank head
  - no bounded side paths

## What Changes From 275a
- keep deterministic autoregressive rollout
- replace direct surface-space GRU dynamics with deterministic latent-state dynamics
- keep the backbone fully deterministic

## Kill Criteria
- if `275b` does not materially improve change-KS / jump-scale / active-cell MR over `275a`, stop extending deterministic latent-state complexity
- if support or cross-cell structure regress badly, the added latent state is not helping the Stage A objective

## Next Step
Implement `275b-v0` in fresh files and evaluate it under the unchanged full 11-suite, while interpreting it primarily as the deterministic 8/11 backbone candidate.
