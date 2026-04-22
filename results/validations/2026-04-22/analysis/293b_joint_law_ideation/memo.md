## 293b ideation

### Context
`293a-v0` validated the new fixed-horizon conditional joint-law family at the mechanism level:
- stochastic spread is alive
- daily change KS is strong
- cross-cell structure is alive

But it failed on the long-horizon path law:
- level KS
- MR
- regime-sensitive width
- pathwise jump realism

So the next move should stay inside the same family and add exactly one mechanism that targets **global path coupling**.

### Candidate approaches

#### 1. Global future latent + daily joint-token decoder
Keep:
- history encoder
- daily joint-token support
- fixed-horizon conditional likelihood objective

Add:
- one global future latent / path summary variable
- posterior during training from full future window
- prior from history only at inference
- every daily token decode conditions on that shared path summary

Why this is attractive:
- narrowest mechanism for global 30-day coupling
- directly targets weak MR / level-law / jump ordering
- keeps `293a`'s strong local daily law intact

Risk:
- posterior collapse if the decoder can already ignore the global latent

#### 2. Future time-cell token grid with non-causal future attention
Replace daily panel tokens with:
- one token per `(horizon, cell)`
- explicit joint future-time/cell attention

Why this is attractive:
- closer to the original `293a` spec
- more expressive future dependence structure

Why not first:
- materially larger architecture jump
- harder to attribute if it fails

#### 3. Coarse-to-fine two-resolution path law
First model:
- coarse future anchors over a few horizon knots

Then condition:
- daily joint tokens on those coarse anchors

Why this is attractive:
- directly encodes long-horizon shape

Why not first:
- more bespoke
- more decomposition assumptions than needed for the immediate bottleneck

### Recommendation
Choose **Approach 1**.

`293b-v0` should be:
- the same `293a` family
- the same daily joint-token support
- one new global path latent conditioning every daily decode

This is the cleanest response to the current diagnosis:
- local stochastic law is already alive
- what is missing is one coherent future-window-level coupling variable

### Proposed 293b-v0
- history encoder -> history context
- posterior network `q(z_path | history, future)` during training
- prior network `p(z_path | history)` at inference
- daily joint-token decoder conditioned on:
  - history context
  - previous sampled daily tokens
  - shared `z_path`
- objective:
  - token NLL
  - plus small KL on `z_path`

### Kill criteria
`293b` is alive only if it preserves `293a`'s local-law wins:
- coverage stays meaningfully nonzero
- cross-cell structure stays in gate
- change KS stays broadly strong

and materially improves at least one path-law category:
- level KS
- MR
- max-jump KS
- regime differentiation

If not, then the next move should be a larger representation shift, not more latent tweaks.
