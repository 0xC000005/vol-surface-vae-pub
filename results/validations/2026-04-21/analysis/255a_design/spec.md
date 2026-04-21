# 255a-v0 Spec

Date: 2026-04-21

## Goal

Run the first paradigm-shift prototype after the continuous common-path families
(`253/254`) were judged structurally capped.

## Hypothesis

The deterministic future path should not be represented as one continuous shared
trajectory plus small side corrections.

Instead, it should be represented as a sparse conditional mixture of learned future
motifs, with only a small bounded residual adapter.

## Architecture

1. `history -> GRU encoder -> h`
2. `h -> motif logits`
3. sparse top-k soft routing over a learned motif bank in **change space**
4. motif mixture gives base deterministic `delta_t`
5. small bounded residual head refines the delta
6. integrate to levels

Motif bank shape:

- `M x T x D`

with:

- `M = 16`
- `T = 30`
- `D = 25`

## Why change-space motifs

The current deterministic miss is mostly about:

- change law
- jump timing / shape
- temporal path geometry

So `255a-v0` should model future **changes**, not future levels.

## Losses

- weighted SmoothL1 on levels
- weighted SmoothL1 on changes
- SmoothL1 on pathwise max absolute change
- terminal loss
- small residual-budget penalty
- routing entropy-floor penalty so top-k routing does not collapse to a single motif immediately

## Kill Criteria

For the first motif prototype:

1. recover at least the `4/11` frontier
2. improve at least one of:
   - `change KS`
   - `max-jump KS`
   - `distributional_fidelity`
3. avoid obvious dense-collapse artifacts
4. keep residual adapter secondary to motif path

## Decision Rule

- if `255a-v0` cannot recover the frontier or materially improve temporal-law metrics,
  the next paradigm should be a hierarchical latent-token family
- if it does recover or improve, continue once more inside the motif-routing family
