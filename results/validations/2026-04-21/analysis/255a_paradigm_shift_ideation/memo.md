# 255a Paradigm-Shift Ideation Memo

Date: 2026-04-21

## Context

The continuous deterministic center-path families have now failed in two different ways:

- `253`: balanced low-rank/common structure better, but plateaued at `4/11`
- `254a/254b`: changed the backbone, but still collapsed into an over-shared output path and regressed to `3/11`

Crucially, `254b` showed that even with:

- healthier loading-rank statistics
- non-collapsed latent factor usage

the generated panel still collapsed into PC1-dominated dynamics.

That makes the next step a **paradigm shift**, not another within-family tweak.

## What Must Change

The next family should stop assuming that the deterministic future can be represented as:

```text
one continuous shared low-rank path
+ small bounded side corrections
```

That geometry is exactly what kept collapsing.

The new family should instead let the model represent **multiple distinct future dynamic motifs**
and choose among them conditionally, without forcing everything through one common path.

## Candidate Families

### Candidate 1: Learned Future Motif Library + Sparse Conditional Routing

Core idea:

- learn a bank of `M` deterministic future motifs
- each motif is a full `(T,D)` future dynamic archetype
- the history encoder outputs sparse routing weights over motifs
- a small bounded residual adapter refines the chosen motif mixture

Form:

```text
history -> encoder h
route(h) -> sparse weights over motif bank
motif mixture -> base future path
small bounded adapter -> refinement
```

Why it is strong:

- directly attacks mode-averaging
- not forced into one shared low-rank trajectory
- still generalizable across factor panels if motifs live in generic `(T,D)` space
- naturally suited to jump timing / event-pattern diversity

### Candidate 2: Hierarchical latent-variable trajectory model

Core idea:

- sample or infer a high-level future regime token first
- then decode a deterministic trajectory conditional on that token

Why not first:

- more moving parts
- harder to cleanly separate deterministic-ceiling work from stochastic calibration

### Candidate 3: Dense temporal backbone with weak output constraints

Why not first:

- too easy to relearn dense coupling
- weakest link to the actual observed failures

## Recommended Paradigm Shift

The most principled next family is:

## `255a = learned future motif library with sparse conditional routing`

### Why this one

It changes the representational geometry in the exact way the evidence now demands:

- away from one continuous common-path attractor
- toward a small set of conditionally selected future dynamic archetypes

This is not just “more capacity”.
It is a different answer to the question:

> what is a future path?

In `255a`, a future path is a sparse mixture of learned motifs, not a single smooth
low-rank trajectory with bounded side corrections.

## 255a-v0 Design

### Structure

1. `history -> encoder h`
2. `h -> motif logits`
3. sparse soft routing over `M` learned motif tensors in latent space or output space
4. decode motif mixture into deterministic `(T,D)` future
5. add a small bounded refinement head

Recommended first version:

- keep it deterministic
- no stochastic residual layer yet
- motif bank size `M = 16` or `32`
- sparse routing with temperature / top-k softmax

### Representation choice

Prefer motifs in a **latent temporal basis** rather than raw `(T,D)` output tensors:

- keeps parameter count controlled
- still avoids the single common-path bottleneck

Concretely:

```text
motif_m in R^(T,H)
decoder(H -> D) with bounded residual adapter
```

This keeps the system generalizable while allowing distinct future dynamic shapes.

## Why this is more principled than another 254 tweak

- `254b` already showed the failure is not just in loading-rank collapse
- the output geometry itself is wrong
- motif routing changes that geometry directly

## Kill Criteria

For `255a-v0`, the first bar is not `11/11`.

It is:

1. recover at least `4/11`
2. materially improve one of:
   - `change KS`
   - `max-jump KS`
   - `distributional_fidelity`
3. avoid obvious dense-coupling collapse
4. preserve generalizable `(B,T,D)` design

If `255a-v0` cannot beat the continuous common-path families on those terms, then the
motif paradigm is weak and the next shift should be a hierarchical latent-token family.

## Decision

Next iteration should be an `experiment`:

## `255a-v0 = deterministic sparse future-motif model`

This is the cleanest paradigm shift available after both `253` and `254` failed for
different reasons inside the same continuous common-path worldview.
