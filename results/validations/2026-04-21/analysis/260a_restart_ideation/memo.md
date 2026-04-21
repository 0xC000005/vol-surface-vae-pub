# 260a Restart Ideation Memo

Date: 2026-04-21

## Context

The `205-258` tree is now archived as:

- baselines
- falsification history
- evidence for the restart constraints

The new active line must be:

- simpler
- more publishable
- more defensible against vanilla baselines
- still valid as a conditional scenario generator for risk use

The restart constraints are now explicit:

1. keep the **generative core vanilla**
2. require **dynamic latent/common state**
3. require **explicit low-rank factor structure**
4. keep only a **small bounded idio path**
5. remain `(B,T,D)`-generalizable across factor panels

## Decision

The first restart family should be:

## `260a`: Minimal Conditional Factor Flow Matching

This is the smallest defensible restart architecture.

It uses:

- **vanilla conditional flow matching** as the generative core
- a **single temporal backbone** over the future horizon
- an **explicit low-rank readout**
- a **bounded idiosyncratic residual**

No:

- latent-token hierarchies
- motif routing
- custom pulse branches
- learned marginal heads
- multi-loss scaffolding
- special regime submodules

## Why Flow Matching First

Choose **flow matching** before diffusion for the first restart baseline because:

- simpler training objective
- simpler sampler
- fewer moving parts to defend
- faster iteration for the same conditional path task

If `260a` fails quickly, the next direct baseline can be the diffusion analogue with
the same structural head, not a new bespoke architecture.

## Modeling Target

Model the **future change path** rather than the future level path.

Reason:

- the main old-family failures were in change law, MR, jump timing, and temporal structure
- levels can be recovered by integrating predicted changes from the last observed level

So the target object is:

```text
P(ΔX_{1:T} | H)
```

with the future levels reconstructed by cumulative summation.

## Minimal Architecture

```text
history H
  -> GRU encoder -> context h

noisy future change path x_t  (flow-matching interpolation state)
  -> input projection
  -> temporal backbone over horizon
  -> factor-state head u_{1:T}

context h
  -> static low-rank loadings Λ(h)

common velocity/change:
  v_common[t] = Λ(h) u_t

idio path:
  v_idio[t] = bounded residual with budget tied to common RMS

final velocity:
  v[t] = v_common[t] + v_idio[t]
```

This keeps the dynamic part in the temporal backbone and the structural bias in the
readout.

## Why This Is The Right First Restart Test

It directly satisfies the new principle:

- vanilla generative core
- minimal structural bias
- no extra architectural stories to defend

It also gives a very clean falsifier:

- if even this minimal conditional factor-FM baseline cannot recover meaningful
  cross-cell structure and deterministic temporal law, then the restart needs either:
  - a stronger temporal backbone, or
  - a diffusion analogue

without ambiguity about whether the failure came from bespoke add-ons.

## Pre-Registered Expectations

`260a` is **not** expected to reach `11/11`.
It is the first clean baseline of the restarted line.

Success criteria for the first run:

1. trains stably end-to-end
2. evaluator integration works cleanly
3. preserves explicit low-rank/common structure
4. lands in a diagnostically useful region relative to archived baselines

Primary comparisons:

- `183c`
- `250ac`
- `251h`
- `258a`

## Kill Criteria

Treat `260a` as a failed first restart baseline if:

1. `corr_ratio < 0.40`
2. `rank_ratio < 0.30`
3. idio dominates common structure
4. training collapses or sampling is numerically unstable
5. it shows no clear mechanism advantage over both `250ac` and `258a`

If killed, move next to:

- `260b`: same structure, but diffusion instead of FM
or
- stronger temporal backbone with the same low-rank head

## Next Action

Implement `260a-v0` now as the first decisive restart experiment.
