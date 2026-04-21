# 258a-v0 Design Spec

## Context

The `257` latent-token VAE family appears capped by decoder mismatch:

- `257c` proved sample identity can matter
- `257d` showed that stronger structure losses over-couple the same decoder family

So `258a-v0` changes the stochastic latent temporal representation itself.

## Hypothesis

A stochastic dual-timescale latent state-space generator can jointly represent:

- slow shared structure
- fast local shocks
- sample-dependent scenario variation

without the token-mean / over-coupling tradeoff seen in `257`.

## Architecture

- history encoder -> conditioning state
- deterministic slow/fast temporal feature towers
- stochastic slow latent state updated over horizon
- stochastic fast latent state updated over horizon
- sampled innovations injected at every step
- low-rank readout from latent state to panel change
- bounded idio residual path

## Training Objective

Keep the first prototype simple:

- ensemble-mean level loss
- ensemble-mean change loss
- pathwise jump loss
- terminal level loss
- multi-sample CRPS-style losses on levels and changes
- low-rank orthogonality regularizer
- small scale regularizer on latent innovation magnitudes

## Pre-Registered Criterion

`258a-v0` is meaningful only if it beats the stochastic `257` line on at least one
of:

- total `n_pass`
- `coverage`
- `conditionality`
- `rank_ratio`
- `cointegration_ratio`

without collapsing `corr_ratio` or producing zero stochastic spread.
