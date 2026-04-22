## 294b ideation

### Context
`294a` established that an explicit continuous scaffold is a real new mechanism
class inside the fixed-horizon joint-law family.

It improved materially over the late support branch on:
- level KS
- calibration
- rank structure
- over-reversion

But it also made the remaining bottleneck clearer:
- the low-frequency cosine-style scaffold is too globally smooth
- it under-controls localized curvature, reversion, and jump timing

So the next step should stay inside the continuous scaffold family while changing
only the scaffold object itself.

### Decision
Next step: `294b-v0`

Keep:
- fixed-horizon one-stage joint-law framing
- explicit continuous scaffold
- residual daily token law around scaffold increments

Change:
- replace the global low-frequency basis scaffold with a **piecewise-linear knot scaffold**

### Mechanism
Represent the 30-day normalized cumulative path by values at a small set of ordered
future knot horizons, then linearly interpolate those knot values to the full 30-day
scaffold.

Suggested knot horizons:
- `1, 4, 8, 14, 21, 30`

Model:
1. encode history
2. predict joint knot values for all cells
3. linearly interpolate to the full scaffold path
4. train the residual daily token law on deviations around the scaffold increment schedule

### Why this is the smallest principled follow-up
It is still:
- continuous
- differentiable
- explicit
- low-parameter

But unlike the global basis scaffold, it can express:
- localized bends
- mid-window reversions
- piecewise slope changes

without returning to:
- support codebooks
- support-use geometry search
- latent scaffold stacks

### Why this matches the pathology
`294a` already showed the family can:
- preserve the local daily law
- keep cross-cell structure alive
- improve level-law fidelity

What it could not do was bend the path sharply enough.

A knot scaffold attacks exactly that issue:
- more local control than cosine basis
- still simpler and cleaner than a learned support bank

### Expected effect
If this diagnosis is right, `294b` should improve at least one of:
- MR ratio
- active-cell slope corr
- worst-cell cointegration ratio
- jump realism

while preserving most of:
- change KS
- level KS gains from `294a`
- cross-cell structure

### Kill criteria
`294b` is alive only if it improves the continuous scaffold tradeoff rather than
just moving it sideways.

Specifically, it should improve at least one of:
- MR ratio
- worst-cell cointegration ratio
- max-jump KS

without giving back both:
- `25/25` change KS
- and the `294a` level-KS gain.

If not, then the `294` scaffold family is likely closer to a local cap and needs a
different path-shape mechanism class.
