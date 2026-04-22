# 266c Targeted Ideation Memo

## Context

`266b-v0` kept the clean `266` reset intact and replaced the single future code with a short latent token path.

That produced a more informative failure than `266a`:
- the **representation side** improved materially
- the **sampling prior** became the clear bottleneck

Evidence:
- reconstruction rank ratio: `0.747`
- reconstruction cointegration ratio: `0.706`
- sampled rank ratio: `0.229`
- sampled cointegration ratio: `0.371`

So the main question is now:

**what is the smallest first-principles change to the latent prior over token paths that can preserve the improved representation structure at sampling time?**

## Candidate Directions

### Option A: Temporally Aware Latent Sequence Diffusion

Keep the same token-path bottleneck, but replace the flattened-vector denoiser with a **sequence-aware denoiser** operating directly on latent tokens.

Minimal implementation:
- latent tokens remain shape `(K, d)`
- timestep embedding is broadcast across tokens
- context is broadcast across tokens
- denoiser is a small standard temporal sequence model over tokens:
  - token MLP + temporal conv stack
  - or token MLP + single self-attention block

Why this matches the mechanism:
- `266b` improved the latent representation precisely because token positions matter
- flattening the token path into one vector throws away that structure at the prior stage
- a sequence-aware denoiser is the smallest way to let the prior respect token order and local token interactions

Why it is still elegant:
- same overall model story as `266b`
- no new side paths
- no hand-designed factor structure
- still standard latent diffusion, just with the right input geometry

### Option B: Standard Latent Transformer Prior Instead of Diffusion

Replace diffusion with a conditional autoregressive or masked token prior over latent tokens.

Why it is plausible:
- naturally sequence-aware

Why it is not the next step:
- changes the generative core itself, not just the denoiser geometry
- adds a larger research jump before testing the simpler fix

### Option C: Keep Flattened Prior and Increase Bottleneck Size

Increase `K` or `d` and hope the prior uses it.

Why it is not the next step:
- does not address the identified failure directly
- likely becomes “more capacity” rather than a cleaner causal test

## Recommended Next Family

**266c-v0: temporal bottleneck latent diffusion with sequence-aware denoiser**

This keeps everything from `266b` except the prior geometry:

- same history encoder
- same future token bottleneck encoder
- same decoder
- same diffusion objective
- same clean first-principles doctrine

Only change:
- denoise latent tokens as a **token sequence**, not as one flattened vector

## Proposed 266c-v0 Denoiser

Smallest clean version:

1. per-token input:
   - noisy token
   - repeated history context
   - timestep embedding
   - token position embedding

2. sequence model:
   - 2-3 residual temporal conv blocks over token axis

3. output:
   - predicted noise token sequence

Why temporal conv instead of attention first:
- simpler
- more stable
- enough to test whether respecting token order fixes the `266b` prior failure

## Pre-Registered Question

Does a sequence-aware latent prior preserve the improved representation structure from `266b` at sampling time?

The specific success direction is:
- hold or improve move-size / jump realism relative to `266b`
- materially improve rank / cointegration / coverage relative to `266b`

## Kill Criteria

Close `266c-v0` quickly if:
- it still collapses rank and cointegration at sampling time to roughly `266b` levels
- it regresses back to `266a`-style move-size suppression
- it needs hand-added structural bias outside the token prior itself to function

## Decision

Proceed with `266c-v0`.

Do not:
- increase bottleneck size first
- reopen low-rank or bounded correction paths
- switch away from diffusion yet

The clean next test is whether the prior simply needs to respect token sequence structure.
