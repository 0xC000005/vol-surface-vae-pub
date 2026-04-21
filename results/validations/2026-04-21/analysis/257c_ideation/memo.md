# 257c Ideation

Date: 2026-04-21

## Context

`257a` introduced the first live stochastic latent family.

`257b` strengthened latent pressure and improved coverage further, but the follow-up
analysis showed that **sample identity still barely matters**:

- token means matter
- token samples matter very little

So the next choice is:

1. `257c`: change the **training objective** so multiple samples matter
2. `258a`: change the **latent prior** while keeping essentially the same objective

## Decision

Choose **`257c` next**, not `258a`.

## Why 257c Before 258a

A richer latent prior is not the first bottleneck.

Current evidence says:

- the decoder already responds more to token means than to token samples
- sample-dependent variation remains tiny even when latent pressure is increased

That means the next missing ingredient is **training signal for scenario identity**,
not prior expressivity alone.

If the objective does not reward meaningful sample-to-sample structure, a richer
prior can still be ignored or reduced to mean-like behavior.

## 257c Sketch

Keep the `257b` architecture family, but add a **multi-sample scenario objective**.

Minimal version:

- draw `M` posterior samples during training
- decode `M` futures
- add a small ensemble scoring term on future changes / levels
- keep the single-sample reconstruction losses and KL terms

Most plausible objective:

- pointwise CRPS-style sample loss on changes and/or levels
- possibly combined with a small pathwise max-jump sample loss

## What 257c Should Test

Can explicit multi-sample training make **sample identity** matter enough to:

- improve coverage / regime coverage
- preserve cross-cell structure
- without further degrading deterministic structure too much

## Why Not 258a Yet

`258a` is still live as a fallback:

- latent flow prior in token space
- or diffusion prior in token space

But that should come **after** one clean test of whether the current latent family
was mostly objective-limited rather than prior-limited.

## Recommendation

Next decisive experiment:

- `257c`: `257b` architecture + multi-sample scenario objective
