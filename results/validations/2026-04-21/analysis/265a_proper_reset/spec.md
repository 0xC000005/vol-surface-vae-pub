# 265a Proper Reset Spec

## Reset Decision

The active `263`/`264` methodology is retired as the main search process.

Reason:
- `263a` to `263c` showed the recurrent low-rank state-space backbone was useful, but the pinv-derived latent target became the blocker.
- `264a` and `264b` shifted the bottleneck from model quality to teacher engineering and sigma-collapse stabilization.
- That is no longer a clean risk-manager scenario-generator program. It is target-mechanics research.

The reset principle is:

**new clean model family, fresh code, single coherent probabilistic story, no retrofitted target engineering.**

## New Active Family

`265a-v0`: end-to-end variational low-rank state-space factor model.

This is a single generative model for:
- deterministic center path
- stochastic scenario spread
- latent temporal dynamics
- cross-cell dependence

It is not:
- a frozen deterministic core plus residual layer
- a pinv teacher
- a posterior teacher retrofit
- a token/motif/router architecture

## Modeling Object

We want a conditional future path law

`p(x_{1:T} | H)`

for a low-rank financial factor panel, where:
- `H` is recent history
- `x_{1:T}` is the future daily change path in the chosen transformed coordinate
- generated scenarios must preserve:
  - low-rank common structure
  - temporal law
  - conditional mean reversion
  - conditional uncertainty

## 265a-v0 Model

### 1. History Encoder

Input:
- normalized history path in the restart coordinate

Output:
- context vector `h`

Use:
- one GRU encoder

### 2. Latent Prior

Latent state:
- `z_t in R^L`, small `L` such as `8`

Prior:
- recurrent latent state-space prior
- `p(z_t | z_{t-1}, h)`

Outputs per horizon:
- latent prior mean `mu_t`
- latent prior scale `sigma_t`

This is the same useful structural lesson from `263`, but now it is part of the full probabilistic model, not a prior tied to an engineered teacher target.

### 3. Latent Posterior

Posterior:
- `q(z_{1:T} | H, x_{1:T})`

Use:
- a small bidirectional temporal encoder over future targets plus repeated history context

Outputs per horizon:
- posterior mean `mu_q_t`
- posterior scale `sigma_q_t`

This is learned jointly from day one. No pinv target. No epsilon-target retrofit.

### 4. Decoder

Decode latent path to panel changes through:
- low-rank common-factor readout
- small bounded idio path
- bounded history-mean EC baseline

The EC baseline is kept only because it was the one clean deterministic mechanism that repeatedly helped MR without requiring bespoke branches.

The decoder remains low-rank and generic over `(B, T, D)`.

### 5. Observation Model

Observation target:
- future path in the transformed change coordinate used by the clean restart line

Use:
- direct reconstruction loss on future path
- no separate residual scenario layer
- no teacher-generated latent targets

## Training Objective

Train end-to-end with one coherent objective:

`L = recon + lambda_kl * KL(q || p) + lambda_level * level_aux + lambda_terminal * terminal_aux + lambda_ortho * ortho`

Where:
- `recon` is reconstruction on future path in the transformed change coordinate
- `KL(q || p)` is per-step posterior-to-prior matching
- `level_aux` and `terminal_aux` are the minimal deterministic auxiliaries retained from the restart line
- `ortho` keeps the low-rank readout identifiable

What is explicitly removed:
- flow-matching target engineering
- pinv latent teacher
- posterior mean-only teacher
- posterior epsilon-only teacher
- residual scenario decomposition

## Retained Components

Retain from the restart line:
- non-AR mainline
- dynamic latent state
- explicit low-rank factor readout
- bounded idio path
- bounded history-mean EC baseline
- transformed change coordinate
- common 11-suite evaluation harness

## Explicitly Banned

The reset line does not allow:
- pinv-derived latent targets
- posterior teacher retrofits as the primary training object
- scale-anchor patches
- token/motif/router branches
- pulse/jump special heads
- frozen-core residual scenario layers
- suite-specific losses
- evaluator-specific hacks

If any future idea requires one of those, it must trigger a new paradigm review first.

## Why This Is More Principled

This family is preferable to continuing `264` because:
- every probabilistic object is in the model specification
- the prior and posterior are trained together
- uncertainty is not a byproduct of target engineering
- the model can be explained end-to-end as a conditional latent state-space model with low-rank readout

That is much easier to defend than “we tried several latent teachers until one stabilized.”

## First Baseline Definition

`265a-v0`:
- GRU history encoder
- recurrent latent prior
- bidirectional posterior encoder
- low-rank decoder
- bounded idio path
- bounded EC baseline
- ELBO-style training only

No FM in `v0`.

If `265a-v0` fails, the next comparison is not another teacher variant. It is whether a vanilla latent FM or latent diffusion core should replace the Gaussian latent prior while keeping the same probabilistic structure.

## Pre-Registered Gates

For `265a-v0` to count as a valid reset baseline, it must:

1. Stay numerically stable end-to-end.
2. Avoid trivial stochastic collapse:
   - `coverage90 >= 0.20`
   - `calibration_error <= 0.35`
3. Preserve structural quality:
   - `corr_ratio` in `[0.70, 1.50]`
   - `rank_ratio` in `[0.70, 1.40]`
4. Preserve useful deterministic quality:
   - `mr_gt_ratio >= 0.50`
5. Produce a posterior/prior scale relationship that is not pathologically inverted.

This is not the final success bar. It is the first reset sanity bar.

## Immediate Next Step

Implement `265a-v0` in fresh files.

Do not import the `264` teacher logic into the new implementation.
Only reuse:
- normalization helpers
- dataset window builders
- evaluation harness integration

Everything else should be a clean implementation of the new probabilistic spec.
