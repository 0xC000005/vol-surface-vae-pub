## 294a ideation

### Context
The `293` family exhausted two local subfamilies:
- support representation search (`293d/293e/293f`)
- support use search (`293g/293h`)

The combined read is:
- the branch can keep local stochastic law, coverage, and cross-cell structure alive
- but it still lacks a mechanism that directly controls **long-horizon path shape**
- stronger scaffold control changes behavior, but does not break the dead suites

So the next move should not be another support variant.

### Decision
Next step: `294a-v0`

Keep:
- fixed-horizon one-stage joint-law framing
- daily residual token law
- explicit path-shape scaffold idea

Change:
- replace the explicit support codebook with a **continuous low-frequency path basis scaffold**

### Mechanism
Represent the 30-day future path by:
1. a small continuous coefficient vector over a fixed low-frequency temporal basis
2. a residual daily token law around the scaffold increments implied by that basis path

Concretely:
- target object for the scaffold:
  - future normalized cumulative path relative to the current state
- basis:
  - fixed low-frequency temporal basis over the 30-day horizon
  - for example low-order DCT / cosine basis
- model:
  - history encoder predicts the basis coefficients jointly for the full panel
  - decode those coefficients into a continuous future scaffold
  - condition the daily residual token law on the scaffold and train it on deviations around the scaffold increment schedule

### Why this is materially different
It is not:
- another hidden latent (`293b/293c`)
- another support codebook (`293d/293e/293f`)
- another support-use geometry tweak (`293g/293h`)

It is:
- an explicit, supervised, continuous low-frequency path controller

So the path-shape object is now:
- differentiable
- continuous
- directly trained against future trajectory shape

That is the smallest new mechanism class that still fits the current family.

### Why this is clean
No:
- retrieval bank
- AR rollout
- extra side branch stack
- custom evaluator logic
- hand-tuned support object codebooks

Just:
- one low-frequency basis scaffold
- one residual daily law around it

### Expected effect
If this diagnosis is right, `294a` should:
- keep more of the local-law behavior than `293g`
- provide stronger explicit path-shape control than `293h`
- improve at least one of:
  - level KS
  - regime width timing
  - jump ordering
  - MR profile

### Kill criteria
`294a` is alive only if it materially improves at least one dead path-shape suite while preserving:
- change KS
- cross-cell structure
- surface validity
- non-degenerate cointegration

If it still lands on the same `4/11` support-branch tradeoff, then the whole `293` fixed-horizon token-law family is likely near a broader local cap.
