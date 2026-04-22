# 266b Targeted Ideation Memo

## Context

`266a-v0` was the first clean first-principles reset baseline:

- fixed-horizon conditional generator
- history encoder
- single compressed future code
- vanilla latent diffusion prior
- no hard low-rank head
- no bounded idio path
- no bounded EC baseline

It trained stably and produced nontrivial spread, but the postmortem showed a clean failure:

1. the **single-vector bottleneck + direct decoder** already over-smooths the future path, suppresses jump scale, and overstates short-horizon mean reversion
2. the **latent diffusion prior** then further collapses the sampled joint law toward a stronger common mode and weaker cointegration

So the active question is no longer “should we reintroduce hand-crafted structure?”

It is:

**what is the smallest first-principles extension beyond a single compressed future code that adds temporal and joint flexibility while keeping the model elegant?**

## Candidate Families

### Option A: Temporal Bottleneck Latent Path Diffusion

Replace the single future code with a **short latent token path**:

- future encoder maps the full future path to `K` latent tokens, with `K << T`
- latent diffusion models the entire token sequence jointly, conditioned on history
- decoder upsamples / decodes the token path back to the full future path

Example:
- `T = 30`
- `K = 5` or `6`
- latent token dim `d = 32`

Why it directly addresses the 266a failure:
- keeps a bottleneck, but makes it **temporally structured**
- allows different parts of the horizon to carry different information
- reduces the burden on one global code to explain both short-horizon dynamics and long-horizon shape
- stays fully generic over factor panels and horizon length

Why it is still elegant:
- same overall story as `266a`
- only one conceptual change: `z` becomes a short latent path instead of a single vector
- still vanilla latent diffusion
- no hard-coded factor/loadings decomposition
- no side correction paths

### Option B: Autoregressive Latent Sequence Generator

Keep the future bottleneck, but generate latent tokens autoregressively instead of with joint diffusion.

Why it is plausible:
- could improve temporal coherence and long-horizon dependence

Why it is not the next choice:
- adds AR rollout assumptions before the fixed-horizon bottleneck family is understood
- less aligned with the direct fixed-horizon evaluation object
- more likely to re-open exposure-bias and long-rollout issues too early

### Option C: Observation-Space Diffusion Without Bottleneck

Drop the future bottleneck and run a direct conditional diffusion model on the future path.

Why it is plausible:
- maximally Bitter-Lesson aligned in one sense

Why it is not the next choice:
- removes the only deliberate compression bias the reset retained
- makes it harder to diagnose what the model is failing to represent versus failing to generate
- increases compute and weakens the “smallest clean extension” discipline

### Option D: Stronger Decoder Only

Keep the single code, but make the decoder deeper or more expressive.

Why it is plausible:
- simplest code change

Why it is not the next choice:
- the failure is not just decoder weakness in the generic sense
- one global code is itself the bottleneck
- increasing decoder power without changing bottleneck structure risks hiding the same representational limitation behind more capacity

## Recommended Next Family

**266b-v0: temporal bottleneck latent path diffusion**

This is the smallest principled extension because it changes exactly one assumption:

- from **single future code**
- to **short future latent path**

Everything else stays aligned with the reset:
- fixed-horizon
- encoder-decoder bottleneck
- vanilla latent diffusion
- no hard low-rank structure
- no bounded side paths
- no suite-specific losses

## Proposed 266b-v0 Shape

- history encoder:
  - same role as `266a`
  - outputs context vector `h`

- future bottleneck encoder:
  - encode future path into `K` latent tokens
  - simplest implementation: temporal striding / learned pooling over future hidden states

- latent diffusion prior:
  - standard diffusion over flattened latent token path `(K * d)`
  - conditioned on history context

- decoder:
  - decode latent token path plus history context into the full future path
  - use a lightweight temporal decoder that can attend to token position

Suggested starting values:
- `K = 5`
- `d = 32`

## Why This Is The Most Principled Next Step

It stays faithful to:
- first principles
- the Bitter Lesson
- elegance
- generalization beyond IV-only 30-day forecasting

It does not assume:
- low-rank structure
- explicit factor decomposition
- mean-reversion pullback
- cellwise correction paths

It only assumes:
- useful future information should compress through a **small temporally structured bottleneck**, not necessarily a single vector

## Pre-Registered Questions

`266b-v0` should answer these questions cleanly:

1. Does a temporal bottleneck reduce the decoder over-smoothing pathology?
2. Does it improve jump-scale realism without adding explicit jump heads?
3. Does it reduce common-mode collapse relative to `266a` by letting different horizon segments carry different latent information?
4. Does it preserve the good part of `266a`:
   - stable training
   - valid support
   - nontrivial stochastic spread

## Kill Criteria

Close `266b-v0` quickly if:

1. it regresses to near-zero spread or unstable training
2. it shows no improvement over `266a` on **both**:
   - rank / cointegration structure
   - jump realism / move-size profile
3. it needs side-path corrections or hand-designed factor heads to function at all

## Decision

Proceed with `266b-v0`.

Do **not** reopen low-rank heads, bounded idio, bounded EC, or teacher engineering.

The next step is to implement `266b-v0` in fresh files and compare it directly against `266a-v0`.
