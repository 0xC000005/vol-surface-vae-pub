## 293c ideation

### Context
`293a` proved that the new fixed-horizon joint-law family can learn:
- strong daily change-law fidelity
- live stochastic spread
- usable cross-cell dependence

`293b` then added one shared global path latent. It improved calibration and cointegration slightly, but it did not fix:
- level KS
- mean reversion
- regime-sensitive width allocation
- pathwise jump ordering

The clean read from the `293a` vs `293b` comparison is that the current family no longer has a generic latent-capacity problem. It has a **path-shape control** problem.

### Decision
Next step: `293c-v0`

Keep:
- the `293a/293b` fixed-horizon joint-token local-law core
- the same joint next-change codebook
- the same history encoder and autoregressive token decoder over the 30-day window

Change:
- replace the single shared global latent with a **small horizon-knot latent scaffold**

### Hypothesis
The missing mechanism is not more entropy. It is **time-structured low-frequency control**.

A single global latent can regularize the full window, but it cannot say:
- early path should lift, then revert
- width should widen more in the middle or tail
- jump allocation should differ early vs late

So `293c` should condition each future day on an interpolated latent scaffold defined by a few coarse horizon knots.

### Proposed mechanism
Use a small set of latent knot states over the 30-day future window.

Minimal version:
- `n_knots = 5`
- knot horizons at approximately weekly anchors:
  - day 1
  - day 7
  - day 14
  - day 21
  - day 30

Model:
1. Encode history as in `293b`.
2. Infer or sample one latent vector per knot.
3. Linearly interpolate those knot latents across the full 30-day horizon.
4. Add the interpolated step latent to each decoder step before token prediction.

This creates a coarse-to-fine factorization:
- coarse scaffold controls low-frequency path shape
- daily token law controls local move realization

### Why this is materially different from 293b
`293b` uses one latent vector for the whole future window.

That latent is constant across all 30 decoder steps, so it can only act as:
- global context
- dispersion regularizer
- weak window-level dependence prior

`293c` instead gives the decoder a **time-varying latent signal**. That is the smallest change that directly targets:
- levels over the window
- mean-reversion timing
- regime width timing
- jump ordering across horizons

### Why this is still clean
This is not a new branch stack.

It adds exactly one mechanism:
- a structured low-frequency latent scaffold

It does not add:
- retrieval
- AR rollout
- deterministic mean branch
- residual transport
- hand-built low-rank constraints
- evaluator-specific losses

### Kill criteria
`293c` is only alive if it materially improves at least one of:
- level KS
- MR ratio / active MR support
- regime width differentiation
- pathwise max-jump KS

while preserving:
- surface validity
- change KS
- cross-cell structure

If it only improves calibration or cointegration again while leaving the path-shape suites dead, then the family is still missing the right long-horizon mechanism.

### Most principled next step
Implement `293c-v0` with:
- knot latent prior/posterior
- interpolated stepwise latent conditioning
- otherwise unchanged `293b` local-law machinery
