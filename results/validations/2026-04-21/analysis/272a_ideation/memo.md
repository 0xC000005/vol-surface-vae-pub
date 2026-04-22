# 272a Paradigm-Shift Memo

## Context

The direct first-principles restart line now has a clear boundary:

- direct observation-space generators (`268`, `269`) failed
- autoregressive latent-token next-step generators (`270`, `271`) improved representation,
  but the per-step token-to-observation decoder stayed too weak and too support-loose

The repeated issue is no longer latent collapse alone. It is that a tiny decoder is
being asked to simultaneously:

- reconstruct the observation manifold
- preserve state-dependent geometry
- and produce the next-step conditional law

That is too much pressure for the current family.

## Goal

Keep the reset doctrine:

- first-principles
- Bitter Lesson aligned
- no hard low-rank structure
- no bounded side paths
- no teacher/KL patch stack

But stop forcing a tiny next-step decoder to learn the observation manifold implicitly.

## Decision

Choose `272a-v0`: autoregressive latent state-space flow matching with a learned
observation autoencoder.

## Core Architecture

Three parts only:

1. `surface_encoder`
   - encode the current normalized surface into a narrow latent state

2. `latent_transition`
   - autoregressive flow-matching transition for the next latent state
   - conditioned on the current latent state

3. `surface_decoder`
   - decode a latent state back into the normalized surface

Training story:

- encode `x_t -> z_t`
- encode `x_{t+1} -> z_{t+1}`
- train latent FM on `z_t -> z_{t+1}`
- train decoder to reconstruct surfaces from latent states

Inference story:

- encode the last observed surface to `z_t`
- sample `z_{t+1}` with the latent FM transition
- decode to `x_{t+1}`
- roll forward autoregressively

## Why This Is More Principled

- It keeps the bottleneck idea, but makes the observation model explicit instead of
  hiding it inside a tiny next-step decoder.
- It uses the minimum state-space decomposition needed to let the model learn:
  - observation manifold
  - latent dynamics
  - scenario uncertainty
- It remains general across financial factor panels and longer horizons.

## What It Does Not Assume

- no low-rank decoder
- no bounded idio path
- no bounded EC baseline
- no explicit copula structure
- no special jump head

## Risk

`272a-v0` is a Markov latent state-space model. If it fails, one likely reason will be
that the latent state itself needs explicit short memory rather than pure one-step
Markov dynamics.

That would still be a clean next diagnosis.

## Next Step

Implement `272a-v0` in fresh files and evaluate whether making the observation model
explicit is enough to beat the exhausted token-decoder line.
