## 293d ideation

### Context
The `293a/293b/293c` comparison makes the next move unusually clear.

What is alive:
- local stochastic move law
- calibration / spread
- cross-cell dependence

What is still dead:
- level-law fidelity
- mean reversion
- regime width timing
- pathwise jump ordering

The latent-conditioning subfamily is now close to capped:
- `293b` used one global latent
- `293c` used a structured knot-latent scaffold
- neither became a true long-horizon path-shape controller

### Decision
Next step: `293d-v0`

Keep:
- the `293` fixed-horizon joint-token local-law core
- the same history encoder
- the same daily joint token codebook and decoder family

Change:
- replace latent path scaffolding with an **explicit coarse path support object**

### Hypothesis
The current family does not just need more context. It needs a path-shape object that is itself in the training target.

Latents are too weak here because they only influence the decoder indirectly.
If the missing suites are:
- level KS
- MR
- regime width timing
- jump ordering

then the model should condition on a representation that directly summarizes:
- where the path sits at a few coarse horizons
- not only on a hidden latent that hopes to imply that structure

### Proposed mechanism
Build a small **coarse path codebook** over future knot-horizon states.

Minimal version:
- choose the same 5 knot horizons used in `293c`
  - day 1
  - day 7
  - day 14
  - day 21
  - day 30
- represent the coarse future shape as the normalized future levels at those horizons
- flatten that `(5 x 25)` knot panel into one coarse path vector
- cluster those coarse path vectors into a modest codebook

Model:
1. Encode history.
2. Predict a distribution over coarse path codes.
3. During training, condition the daily token decoder on the teacher coarse path code.
4. During sampling, sample one coarse path code first, then decode the daily token sequence conditioned on that explicit coarse support object.

### Why this is materially different from 293b/293c
`293b` and `293c` both use hidden latents.

`293d` instead uses an **explicit coarse support object** that corresponds directly to low-frequency future trajectory shape.

So `293d` is not:
- more latent size
- more latent time structure
- another KL tweak

It is:
- a direct path-shape support representation

### Why this is still clean
This adds one mechanism:
- a coarse future path code

It does not add:
- retrieval
- AR rollout
- deterministic/stochastic two-stage decomposition
- custom evaluator hacks
- low-rank side structure

The local daily token law remains the same.

### Risks
The main risk is that a coarse discrete path code may become too rigid and hurt:
- change KS
- local spread
- cross-cell structure

So `293d` is only justified if it clearly improves at least one path-shape suite without giving back the current family wins.

### Kill criteria
`293d` is alive only if it materially improves at least one of:
- level KS
- MR
- regime differentiation
- pathwise jump realism

while preserving:
- change KS
- coverage
- cross-cell structure

If it only improves calibration or cointegration again, then the coarse support object is not the right path-shape representation either.

### Most principled next step
Implement `293d-v0` as:
- history encoder
- coarse path codebook head at knot horizons
- daily joint-token decoder conditioned on the coarse code

Keep everything else from `293a/293b/293c` fixed.
