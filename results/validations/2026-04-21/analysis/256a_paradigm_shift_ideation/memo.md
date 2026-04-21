# 256a Paradigm-Shift Ideation

Date: 2026-04-21

## Context

The autoresearch loop has now falsified three deterministic future-representation
families:

- `253`: continuous dynamic factor state-space family
- `254`: dual-timescale continuous common-path family
- `255a`: fixed sparse motif-routing family

The failure modes differ:

- `253`: better structure, but hard temporal-law / KS tradeoff
- `254`: over-shared common mode and PC1-style collapse
- `255a`: near-single-motif collapse despite a sparse routing design

So the remaining problem is not just “choose better losses.” It is that the
**future representation itself** is too rigid when it is forced into either:

- one continuous shared path, or
- one fixed motif library

## New Paradigm

Move to a **hierarchical latent-token future representation**.

Working name:

- `256a`: deterministic hierarchical future-token model

## Core Idea

Represent the 30-day future path with a small set of **history-conditioned latent
tokens**, not with:

- a single common continuous factor path
- or a fixed bank of reusable motifs

The tokens are generated fresh for each history window, so the basis is adaptive
rather than fixed.

## 256a-v0 Sketch

1. `history -> encoder -> context h`
2. `h -> one slow global token`
3. `h -> K fast event tokens`
4. learned temporal queries for each future step attend to the token set
5. attention outputs a latent future state per horizon
6. low-rank readout converts those states to `delta_t`
7. small bounded residual adapter refines the path
8. integrate deltas to levels

## Why This Is More Principled Than Another 255 Variant

`255a` failed because the motif library itself collapsed and the router then selected
that collapsed basis almost deterministically.

`256a` removes the fixed library bottleneck:

- the token set is produced from the history window
- multiple tokens can contribute to different horizons
- the decoder can represent both slow structure and localized events
- the readout can stay low-rank without forcing one global path

This directly targets the shared weakness across `253/254/255`:

- all three effectively reduced the future to one dominant basis element

## Constraints To Keep

- non-AR
- fixed-horizon 30-day generation
- generic `(B,T,D)` internal representation
- low-rank or otherwise controlled output coupling
- bounded residual side path
- deterministic only for `v0`

## Kill Criteria For 256a-v0

1. recover at least the `4/11` deterministic frontier
2. improve at least one of:
   - `change KS`
   - `max-jump KS`
   - `distributional_fidelity`
3. avoid token collapse:
   - more than one token must be materially used across validation windows
4. avoid `254`-style over-shared common-mode collapse:
   - no severe PC1 domination in the output path

## Decision

`256a` is the most principled next family.

If `256a-v0` cannot recover the `4/11` frontier or collapses into a single-token
future representation, the next shift should be away from deterministic token
representations entirely and toward a richer latent generative family.
