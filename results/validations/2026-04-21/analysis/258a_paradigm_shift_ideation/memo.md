# 258a Paradigm Shift Memo

## Context

The `257` latent-token VAE family is now likely capped.

Evidence:

- `257a`: first live stochastic family, `3/11`
- `257b`: stronger latent usage, but still token-mean dominated
- `257c`: multi-sample objective proved sample identity can matter
- `257d`: best objective-only structure-preserving follow-up still failed and pushed
  the decoder into an over-coupled common mode

So the remaining bottleneck is no longer "dead stochasticity". It is the **decoder
family itself**:

- too static
- too token-mean dominated
- too easy to over-couple under stronger structure losses

## Decision

Shift to **`258a`**, a stochastic dual-timescale latent state-space generator.

## Why Not a Richer Token Prior

The old `258a` fallback was "richer latent prior in token space". That is no longer
the best next move.

Reason:

- `257c` and `257d` show that the issue is not only prior expressivity
- the current token decoder/attention/readout family is the thing that saturates
- making the prior richer while keeping the same decoder is likely to reproduce the
  same token-mean vs over-coupling tradeoff

So the paradigm shift should change the **latent temporal representation**, not just
the prior over static future tokens.

## 258a Family

### Core Idea

Use a **stochastic latent state-space model** with separate slow and fast latent
states that evolve over the future horizon.

- history encoder initializes latent state
- sampled innovations drive future stochasticity over time
- low-rank readout maps latent state to the panel
- bounded idiosyncratic residual path remains secondary

### Why This Fits the Evidence

It directly addresses the `257` family failure modes:

1. **Static token bottleneck**
   - replace static token bank with evolving latent state path

2. **Token-mean dominance**
   - stochasticity enters recurrent latent dynamics directly, not only token choice

3. **Over-coupling under structure losses**
   - use explicit low-rank latent states plus bounded idio path instead of asking a
     static decoder to satisfy both spread and structure

4. **Need both slow structure and fast shocks**
   - slow state handles rank / long-run dependence / mean reversion
   - fast state handles jump timing / local scenario variation

## 258a-v0 Sketch

- history encoder -> `h0`
- latent states:
  - `z_slow_t`
  - `z_fast_t`
- stochastic innovations:
  - `eps_slow_t`
  - `eps_fast_t`
- latent transitions:
  - `z_slow_t = f_slow(z_slow_{t-1}, h0, eps_slow_t)`
  - `z_fast_t = f_fast(z_fast_{t-1}, z_slow_t, h0, eps_fast_t)`
- readout:
  - low-rank factor head from `[z_slow_t, z_fast_t]`
  - bounded idio residual head
- output:
  - future change path, then integrate to levels

## Training Principle

Keep the first `258a-v0` objective simple:

- deterministic level/change/jump losses
- multi-sample scenario losses
- hard idio budget

Do **not** start with diffusion or flow in latent space.

First ask whether a dynamic stochastic latent state-space family is already enough to
break the token-decoder ceiling.

## Pre-Registered Success Criterion

`258a-v0` is a meaningful shift only if it beats the stochastic `257` frontier on at
least one of these without collapsing structure:

- total `n_pass`
- `conditionality`
- `coverage`
- `rank_ratio`
- `cointegration_ratio`

## Recommendation

Next decisive experiment:

- `258a-v0`: stochastic dual-timescale latent state-space generator with low-rank
  readout and bounded idio path
