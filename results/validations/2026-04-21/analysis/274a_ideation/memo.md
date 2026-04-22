# 274a Ideation

## Why 273a Is Closed
- `273a` kept the useful explicit observation model, and its token autoencoder reconstructs real future surfaces without support explosions.
- The failure appears in the sampled latent transition itself: FM on deterministic token-state deltas leaves the learned latent manifold on the first step.
- That means more deterministic delta-stabilization is the wrong response. The problem is not a missing scale knob; it is that the latent manifold is not part of the probabilistic model.

## Next Family
`274a-v0`: probabilistic latent-token state-space model with explicit observation model

## Core Model
- Observation encoder: future surface -> latent token posterior parameters
- Observation decoder: latent tokens -> surface
- Latent prior / transition: previous latent tokens -> next latent-token prior parameters
- Autoregressive rollout by sampling next latent tokens from the learned prior, then decoding

## What It Keeps
- explicit learned observation model
- small latent token set
- sequence-aware token transition
- no hard low-rank head
- no bounded side paths
- no teacher/target-engineering machinery

## What It Removes
- deterministic FM on raw autoencoder latent deltas
- assumption that a reconstruction-only latent manifold is generative enough for sampling

## Minimal Training Story
- posterior: `q(z_t | x_t)`
- prior: `p(z_t | z_{t-1})`
- decoder: `p(x_t | z_t)`
- optimize a simple autoregressive ELBO:
  - reconstruction loss on `x_t`
  - KL(q || p) for adjacent steps
- no KL warmup, free bits, auxiliary routers, or handcrafted stabilizers in `v0`

## Why This Is First-Principles Enough
- This is the minimal standard generative fix for the exact pathology observed in `273a`: if generated latent states leave the useful manifold, the manifold itself must be learned as part of the generative model.
- It stays general across financial factor panels and longer horizons because it only assumes a bottlenecked latent state and autoregressive conditional generation.

## Pre-Registered Kill Criteria
- if KL collapses to ~0 and prior/posterior std remain indistinguishable, close the family quickly
- if reconstruction remains on-manifold but sampled one-step latent rollout still explodes at step 1, the latent prior is too weak
- if `274a` does not beat `273a` on surface validity and distributional fidelity together, do not stack posterior tricks; reconsider the family

## Next Step
Implement `274a-v0` in fresh files and run the full train/eval/postmortem loop.
