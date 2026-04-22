# 270a Ideation Memo

## Context

The strict first-principles direct-generator reset is now closed:
- `268a`: direct level-space FM was alive but weakly conditioned.
- `268b`: stronger history conditioning did not fix the family; support got worse.
- `268c`: direct change-space FM fixed temporal moments but lost state-dependent mean reversion.
- `269a`: autoregressive direct next-change FM recovered some conditionality, but blew up width and cross-cell structure.

The clean common lesson is that pure observation-space generators are too loose. Some learned shared bottleneck is necessary.

## Constraint Set

The next family must still respect the reset bans:
- no hard low-rank decoder
- no bounded idio path
- no bounded EC baseline
- no teacher-engineering / KL hacks as the core story
- no suite-specific loss stack

## Recommended Next Family

**270a-v0: autoregressive latent-bottleneck next-change flow matching**

Minimal story:
- encode the rolling history into a narrow latent state
- sample next-step latent innovation with vanilla FM in latent space
- update latent state autoregressively
- decode next normalized level from latent state
- feed the generated level back into the next step

This restores one structural bias only:
- a narrow learned bottleneck

It keeps the two lessons from `269a`:
- state feedback matters
- pure observation-space generation is too loose

## Why This Is Principled

- The bottleneck is generic and data-driven, not a hand-coded factorization.
- The model remains a conditional scenario generator over future paths.
- The autoregressive rollout gives the model the evolving state dependence that `268c` lacked.
- The latent bottleneck gives the model a shared hidden state so it does not diffuse into the over-wide, over-rank direct-generator failures.

## Minimal 270a-v0 Design

- history encoder: GRU or causal temporal encoder into narrow state `z_t`
- latent transition: FM over next-step latent increment
- decoder: MLP from latent state to next normalized level
- training: one-step autoregressive teacher forcing over future horizon
- sampling: rollout recursively for 30 days

## Pre-Registered Question

Can a narrow learned bottleneck recover:
- state-dependent mean reversion
- usable cross-cell structure
- non-explosive width
without reintroducing hand-engineered finance structure?

## Decision

Proceed to `270a-v0` implementation.
