# 273a Ideation Memo

## Context

`272a` and `272b` established a clean pattern:

- explicit observation modeling is helpful
- a single latent vector state is too compressive and too globally coupled
- adding GRU memory on top of the same single latent vector does not fix that

So the next family should change the **latent state structure**, not just the transition
conditioning around the same scalar latent bottleneck.

## Constraint

Keep:

- first-principles simplicity
- explicit learned observation model
- autoregressive latent dynamics
- no hard low-rank head
- no bounded side paths
- no teacher/KL machinery

## Decision

Choose `273a-v0`: latent token state-space flow matching with explicit observation model.

## Core Architecture

Three learned pieces only:

1. `surface_encoder`
   - maps current normalized surface to a small set of latent tokens

2. `token_transition`
   - flow-matching transition over the latent token set
   - sequence-aware across tokens (self-attention or transformer block)

3. `surface_decoder`
   - reconstructs the normalized surface from the latent token set

## Why This Is More Principled

- It keeps the observation model explicit, which was the main gain of `272a`.
- It changes the latent state itself, which is now the clean bottleneck.
- A token state is still general and learned from data; it does not impose hard
  low-rank or hand-designed factor structure.
- It is the smallest architectural move that can plausibly reduce the over-common-mode
  collapse seen in `272a/272b`.

## Rejected Alternatives

### More recurrence on the same single latent vector

Rejected because `272b` already falsified that.

### Immediate transformer over raw observation history

Rejected for now because that jumps past the useful gain of the explicit observation
model and reopens a broader design space too early.

## Kill Criteria

`273a` is worth keeping only if it improves at least one of:

- `corr_ratio` / `rank_ratio`
- aggregate and long-horizon mean reversion
- regime-sensitive width allocation

without giving back the manifold gains from `272a`:

- calendar / butterfly validity
- cellwise MAE
- cointegration

## Next Step

Implement `273a-v0` in fresh files and test whether a structured latent token state is
enough to escape the single-vector collapse while preserving the cleaner observation
model.
