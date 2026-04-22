## 293e ideation

### Context
The `293b/293c/293d` bracket is now clean enough to support one narrow next move.

What is capped:
- hidden latent conditioning for path shape

What is alive:
- explicit coarse path support

What is weak in the current `293d` implementation:
- one monolithic coarse path code over the full `(5 x 25)` knot panel
- low coarse-code accuracy
- all-or-nothing prediction difficulty

### Decision
Next step: `293e-v0`

Keep:
- the `293` daily joint-token local-law core
- the explicit coarse path support idea
- the same knot horizons

Change:
- replace the monolithic coarse path code with a **sequence of coarse knot tokens**

### Hypothesis
The support idea is right, but the current support representation is too entangled.

Instead of asking the model to classify one full future coarse path object, ask it to predict:
- one coarse token per knot horizon

This is still explicit path support, but it is now compositional.

### Proposed mechanism
Use a shared or small per-knot codebook over normalized future levels at the knot horizons.

Minimal version:
- 5 knot horizons
- one discrete support token per knot
- token values correspond to the full 25-cell future level panel at that horizon

Model:
1. Encode history.
2. Predict the knot-token sequence, ideally with a small autoregressive coarse decoder over knots.
3. Convert the predicted knot tokens into knot-level panels.
4. Interpolate those knot panels across the 30-day window.
5. Condition the daily token decoder on that interpolated explicit coarse scaffold.

### Why this is better than 293d
`293d` predicts one code over the full coarse path object.

That makes the classifier solve:
- all horizons
- all cells
- all path-shape interactions

at once.

`293e` instead makes the support object compositional:
- the model only has to predict one knot panel at a time
- the coarse scaffold can still express a path
- the codebook prediction problem is easier and less entangled

### Why this is still clean
This does not change the local joint-law core.

It only changes the explicit support representation from:
- monolithic code
to:
- compositional knot-token sequence

No:
- latent branch return
- AR path rollout
- retrieval bank
- custom evaluator hacks

### Kill criteria
`293e` is alive only if it materially improves at least one of:
- level KS
- MR / active support
- regime differentiation
- pathwise jump realism

while preserving:
- change KS
- coverage
- cross-cell structure

If it only improves support-token accuracy without moving the path-shape suites, then the compositional support representation is still not enough.

### Most principled next step
Implement `293e-v0` with:
- coarse knot-token sequence prediction
- interpolated explicit scaffold from those predicted knot tokens
- unchanged daily joint-token local-law decoder
