# 257a Paradigm-Shift Ideation

Date: 2026-04-21

## Context

The autoresearch loop has now exhausted the deterministic center-path families:

- `253`: dynamic continuous factor family capped at `4/11`
- `254`: dual-timescale continuous family regressed through over-shared common mode
- `255a`: fixed motif family regressed through single-motif collapse
- `256a`: hierarchical token family regressed through perfectly uniform dead attention

The repeated lesson is that deterministic basis-selection alone is no longer the
right search space.

## New Paradigm

Move to a **conditional latent generative future-token model**.

Working name:

- `257a`: conditional latent future-token VAE

## Core Idea

Instead of trying to decode the entire future from a deterministic basis selected
from history, learn a **stochastic latent set of future tokens**:

1. `history -> context encoder -> h`
2. `future (train only) -> posterior encoder -> q(z_tokens | history, future)`
3. `history -> prior network -> p(z_tokens | history)`
4. sample latent future tokens
5. temporal queries attend to the sampled latent tokens
6. decoder outputs the full future path

At inference time:

- sample latent token sets from the conditional prior
- decode multiple futures directly

## Why This Is More Principled Now

This directly addresses the last three deterministic failures:

- deterministic bases kept collapsing to one dominant mode
- deterministic token attention in `256a` stayed uniformly dead
- `11/11` ultimately requires stochastic conditional scenario generation anyway

So rather than trying to get to `8/11` with an increasingly contorted deterministic
decoder, move to a family that can learn:

- adaptive future structure
- multiple plausible futures
- conditional uncertainty

in one coherent latent representation.

## Why VAE First

Start with a VAE-style latent generative family before a latent flow/diffusion.

Reason:

- smallest viable stochastic paradigm shift
- easier to train and falsify quickly
- leverages the repo's VAE background
- can later be upgraded to richer latent priors if the Gaussian latent is too weak

## 257a-v0 Sketch

- latent token count: small (`K=4` or `K=6`)
- token dimension: moderate (`64`)
- deterministic token decoder borrowed from the `256a` idea
- but tokens are now sampled from a conditional latent distribution rather than
  produced deterministically from history
- train with:
  - reconstruction losses on levels / changes / jumps
  - KL regularization
  - light anti-collapse checks on posterior token usage

## Kill Criteria

1. recover at least the `4/11` frontier
2. beat deterministic coverage from `0%` in a meaningful way
3. avoid posterior collapse:
   - KL not vanishing to zero
   - sampled token sets materially affect decoded futures
4. preserve or improve at least one deterministic temporal-law suite

## Decision

`257a` is the most principled next family.

If `257a-v0` cannot beat the deterministic frontier or collapses posterior usage,
the next shift should escalate from VAE-style latent tokens to a richer latent flow
or diffusion prior in token space.
