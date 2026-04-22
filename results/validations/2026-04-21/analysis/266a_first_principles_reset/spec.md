# 266a First-Principles Reset Spec

## Reset Decision

`265a` was a transitional reset, not a thorough one.

It still carried forward inherited design commitments from the previous search tree:
- hard low-rank decoder
- bounded idio path
- bounded history-mean EC baseline
- variational prior/posterior language as the default framing

Those may all be useful later. None of them are first-principles requirements.

The stricter reset principle is:

**start from a minimal conditional scenario generator whose only core structural bias is a narrow latent bottleneck.**

Everything else must be re-earned from failure analysis, not inherited because it helped an older family.

## New Active Family

`266a-v0`: first-principles conditional bottleneck generator.

This family is defined by:
- a history-conditioned encoder-decoder structure
- a narrow latent bottleneck where real compression happens
- a vanilla generative core in latent space
- no hard-coded factorization, correction paths, or suite-specific mechanisms in the core model

The family is intended to learn the conditional future path law

`p(x_{1:T} | H)`

with as little hand-engineered structure as possible.

## Modeling Object

We want a valid conditional scenario generator for a future path `x_{1:T}` given recent history `H`.

The model must learn:
- conditional center path
- stochastic scenario spread
- temporal law
- cross-sectional dependence

The reset assumption is:

**if the model fails, the first question should be what the bottleneck generator does not learn by itself, not which hand-designed correction we forgot to add.**

## Core First-Principles Commitments

These are the only core commitments in `266a-v0`:

1. **Conditional history context**
   - the model may use recent history `H`
   - no domain-specific handcrafted summary variables are required

2. **Narrow latent bottleneck**
   - future-path information must pass through a compressed latent representation
   - the compression itself is the only deliberate inductive bias in the core model

3. **Vanilla generative core**
   - the latent generator should be a standard conditional latent diffusion or standard conditional latent flow-matching model
   - no custom teacher targets
   - no residualized two-stage decomposition

4. **Generic decoder**
   - decode latent samples back to the future path directly
   - no hard low-rank readout
   - no bounded side paths in the core spec

## Explicitly Not Assumed In The Core Spec

`266a-v0` does **not** assume:
- hard low-rank structure
- bounded idio path
- bounded EC baseline
- explicit factor/loadings parameterization
- residual scenario layer on top of a frozen deterministic core
- latent teacher engineering
- router/token/pulse/motif branches
- suite-specific losses or evaluator-specific hacks

If any of those return later, they must return only as clearly motivated ablations after the pure bottleneck baseline is understood.

## Why This Is More Principled

This reset is closer to the Bitter Lesson because:
- it removes inherited architectural opinions that were not strictly necessary
- it keeps the model class broad and learnable
- it asks the model to discover temporal and cross-sectional structure through a high-capacity latent generator rather than through hand-imposed factorization

This reset is closer to first principles because:
- the scenario-generator objective is stated directly as conditional future-path generation
- the model family is not defined by fixes to previous pathologies
- failures will be interpretable as learning failures of the core generator, not as interactions between multiple corrective mechanisms

## 266a-v0 Baseline

The first baseline should be:

- direct fixed-horizon conditional generator
- one history encoder
- one future encoder for training-time bottleneck formation
- one future decoder
- one vanilla latent generator conditioned on history

The recommended initial instantiation is:

**conditional latent diffusion in bottleneck space**

because it is standard, easy to justify, and does not require any special target construction beyond the usual diffusion objective.

That means:
- train an encoder-decoder bottleneck for future paths
- learn a conditional latent diffusion model over encoded future latents given history
- sample latent scenarios from the diffusion model at inference
- decode them directly to future paths

This is a clean conditional scenario generator story.

## Why Fixed-Horizon First

`266a-v0` should start as a fixed-horizon multi-day generator.

Reason:
- the evaluation object is a fixed future path
- this is the simplest direct mapping from history to scenario set
- it avoids adding autoregressive rollout assumptions before the pure bottleneck family is tested

This is not a claim that AR is banned forever.
It is only the cleanest first baseline.

## Training Story

Training should be expressed in the most standard way possible for the chosen latent generator:

- reconstruction term for the bottleneck autoencoder
- standard latent diffusion objective
- optional light latent regularization only if needed for bottleneck stability

No:
- custom posterior teacher
- pseudoinverse latent target
- EC pullback term
- bounded per-cell correction path
- bespoke risk-suite shaping losses in the first baseline

## Pre-Registered Sanity Gates

`266a-v0` counts as a valid first-principles baseline only if it:

1. trains stably end to end
2. produces nontrivial stochastic spread:
   - e.g. `coverage90 >= 0.20` or another clearly non-collapsed diversity signal
3. preserves the basic object:
   - valid full 11-suite evaluation
   - no post-hoc correction logic
4. remains architecturally clean:
   - no added low-rank/factor head
   - no bounded idio side path
   - no bounded EC mechanism

The purpose of `266a-v0` is not immediate `11/11`.
The purpose is to establish the cleanest baseline from which later learning failures can be analyzed honestly.

## Immediate Next Step

Implement `266a-v0` in fresh files.

Reuse only:
- data normalization helpers
- window builders
- evaluation harness integration

Do not import:
- `263`/`264` teacher logic
- `260` deterministic correction logic
- `261` residual scenario logic
- `265a` low-rank / bounded-side-path assumptions

If `266a-v0` fails, analyze what the bottleneck generator does not learn before introducing any extra structure.
