## 295a ideation

### Context
The fixed-horizon one-stage joint-law program has now tested three path-shape mechanism
classes:
- latent conditioning (`293b`, `293c`)
- explicit support objects (`293d` through `293h`)
- explicit continuous scaffolds (`294a`, `294b`)

What is now clear:
- the family can learn a strong local stochastic law
- it can keep non-degenerate cross-cell structure alive
- it can improve either structural anchoring (`293d`) or level-law fit (`294a`)

What is not clear:
- how to allocate short-horizon uncertainty, reversion, and jump timing
  **without** collapsing either local-law sharpness or long-window path shape

So the next step should not be another support-object tweak or another scaffold-object
swap.

### Decision
Next step: `295a-v0`

Keep:
- fixed-horizon one-stage joint conditional-law framing
- explicit future-time path-shape control
- daily residual joint law over normalized changes

Change:
- replace explicit support/scaffold objects with a **coarse future state-sequence controller**

### Mechanism
Predict a short sequence of future control states at a small set of knot horizons, then
interpolate or broadcast those control states across daily future steps and let the
daily joint-law head condition on them.

Concretely:
1. encode history
2. decode a short ordered sequence of future control states at knot horizons
3. interpolate those control states across the 30-day window
4. condition the daily joint token law on the interpolated control-state trajectory

Important distinction:
- the control states are **not** direct path levels
- and they are **not** support codebook IDs

They are learned future-time conditioning states for the daily law.

### Why this is materially different
It is not:
- another hidden global latent (`293b`)
- another time-structured latent scaffold (`293c`)
- another monolithic support object (`293d`-`293f`)
- another direct scaffold geometry (`293g`, `293h`, `294a`, `294b`)

It is:
- a compositional **future control-state sequence**
- local in time
- continuous
- but not forced to be a direct path object

This is the smallest clean bridge between:
- the structural anchoring gains from explicit support
- and the level-law gains from continuous scaffolds

### Why this matches the pathology
The current bottleneck is no longer "we need more future context."
It is:
- the daily law needs a future-time-local controller
- but direct path objects either over-anchor or over-smooth

A future control-state sequence attacks exactly that:
- more localized than a global scaffold
- more compositional than a monolithic support code
- less restrictive than directly supervising future levels

### Expected effect
If this diagnosis is right, `295a` should improve at least one of:
- h1 / h7 coverage
- MR ratio
- active-cell MR corr
- max-jump KS

while preserving most of:
- change KS
- cross-cell structure
- the level-law gains unlocked by `294a`

### Decision rule
Continue the fixed-horizon joint-law paradigm for exactly this bounded mechanism-class
shift.

If `295a` fails to jointly improve short-horizon allocation while keeping the current
family's local-law strengths, then the whole fixed-horizon one-stage joint-law paradigm
should be treated as near a broader cap and a paradigm shift should follow.
