# 266d Targeted Ideation Memo

## Context

`266c-v0` kept the `266` reset clean and changed only the latent diffusion prior geometry.

That result clarified the failure:
- the temporal bottleneck representation is still alive
- token-order-aware diffusion improved sampled temporal moments
- but the sampled law still collapses toward excessive common mode, low rank, weak cointegration, and poor calibrated coverage

So the active question is no longer:

**how should the diffusion denoiser be shaped?**

It is now:

**is latent diffusion itself the wrong generative core for this bottlenecked future-token manifold?**

## Candidate Directions

### Option A: Sequence-Aware Latent Flow Matching

Keep everything from `266c` except the generative core:
- same history encoder
- same future token bottleneck encoder
- same decoder
- same token-path latent geometry
- replace latent diffusion with **vanilla conditional flow matching** on the latent token path

Why this is the cleanest next test:
- changes exactly one assumption: the generative core
- stays fully first-principles
- keeps the temporal bottleneck representation fixed
- avoids more diffusion-specific denoiser tuning

Why it may help:
- direct transport may preserve the latent path manifold better than repeated reverse-diffusion denoising
- it is a more direct test of whether the bottleneck representation is usable by a standard conditional generative engine

### Option B: Conditional Autoregressive Prior Over Latent Tokens

Use the same bottleneck, but replace the prior with a small AR model over latent tokens.

Why it is not the next step:
- introduces a stronger factorization assumption over token order
- less aligned with the fixed-horizon scenario objective
- more likely to blur representation and prior effects

### Option C: Direct Path-Space Flow Matching Without Latent Bottleneck

Discard the bottleneck and model future paths directly.

Why it is not the next step:
- that would abandon the one clean bias still supported by `266a/b/c`
- too large a paradigm jump before the simpler core swap is tested

## Recommended Next Family

**266d-v0: temporal bottleneck latent flow matching with sequence-aware velocity field**

Keep the `266c` architecture intact except for one thing:
- replace the latent diffusion prior with a **sequence-aware latent flow-matching prior**

Smallest clean version:
- standard Gaussian latent token path as base distribution
- conditional flow matching from noise token path to encoded future token path
- sequence-aware velocity network with the same broad capacity scale as the `266c` denoiser
- fixed small Euler sampler at evaluation

## Why 266d Is The Most Principled Next Step

`266c` already showed that more denoiser geometry inside diffusion is not the important question anymore.

So the clean next falsifier is:

if the temporal bottleneck representation is real, does a simpler vanilla generative core preserve it better than diffusion?

That is exactly what `266d` tests.

## Pre-Registered Question

Can a vanilla conditional latent flow-matching prior over token paths:
- preserve more of the `266c` reconstruction rank and cointegration at sampling time
- without sacrificing the temporal-moment gains `266c` recovered?

## Kill Criteria

Close `266d-v0` quickly if:
- rank ratio still stays near `266b/266c` collapse levels
- cointegration still stays below `0.50`
- coverage and calibration do not materially improve relative to `266c`
- or the line starts requiring diffusion-style rescue knobs in the FM formulation

## Decision

Proceed with `266d-v0`.

Do not:
- reopen bounded idio or EC paths
- add low-rank heads
- add posterior teachers
- keep tuning diffusion inside the same family first

The clean next test is a **vanilla generative-core swap**, not a structural patch.
