## 293b / 293c / 293d comparison

### Context
`293b`, `293c`, and `293d` are now the clean mechanism bracket for the current `293` family:

- `293b`: one shared global latent
- `293c`: time-structured knot latent scaffold
- `293d`: explicit coarse path support object

The key question is no longer whether the family wants "more context."
It is:
- what kind of path-shape representation actually moves the right suites?

### Score trajectory
- `293b`: `4/11`
- `293c`: `3/11`
- `293d`: `4/11`

So `293d` did not break the frontier, but it recovered the branch after the `293c` regression.

### Stable invariants across all three
Across all three models, the following remain alive:
- coverage
- daily change KS
- cross-cell correlation structure

Numerically:
- coverage90: `0.895 -> 0.892 -> 0.909`
- change KS pass: `23/25 -> 23/25 -> 25/25`
- corr ratio: `1.256 -> 1.244 -> 1.301`
- rank ratio: `0.879 -> 0.881 -> 0.849`

So the local stochastic joint law remains robust.

### What hidden latents could not do
`293b` and `293c` changed:
- calibration
- a bit of cointegration
- a bit of jump KS

But they did **not** change:
- level KS
- regime differentiation
- active mean-reversion support

That established that latent conditioning was not turning into path-shape control.

### What explicit support recovered
`293d` is different in one important way.

Relative to `293c`, it improved:
- score: `3/11 -> 4/11`
- change KS: `23/25 -> 25/25`
- cointegration ratio: `0.610 -> 0.656`
- worst-cell cointegration ratio: `0.158 -> 0.289`
- MR ratio: `0.057 -> 0.093`
- active-cell MR corr: `0.523 -> 0.773`

That is the first time this branch improved **structural path-shape-adjacent metrics** rather than only calibration-like metrics.

### What explicit support still did not solve
Even with `293d`, the hard suites stayed dead:
- level KS: `1/25`
- active MR support rate: still `0`
- regime differentiation: `0.984`
- pathwise max-jump KS: `0.647`

So the support object helped, but it was still too weak to become a decisive trajectory controller.

### Why 293d is still the right direction
The coarse-code head in `293d` was weak:
- early coarse accuracy around `0.10 - 0.15`
- then lower later

But despite that, the model still recovered:
- cointegration
- some MR structure
- perfect daily change KS

That means the support object itself is carrying useful signal even in a weak implementation.

So the right read is:
- hidden latents are near a local cap
- explicit support is directionally right
- the current **monolithic coarse codebook** is probably too entangled and too hard to predict cleanly

### Mechanism conclusion
The bottleneck is now likely:
- not the existence of a support object
- but the **form** of the support object

`293d` predicts one code over the entire `(5 x 25)` coarse future knot panel.
That is a hard classification problem and likely too entangled.

The next clean move should therefore strengthen the explicit support object by making it **more compositional**, not by changing the daily token law again.

### Decision implication
The next mechanism should be:
- explicit support again
- but easier and more structured to predict

Most likely:
- a **coarse knot-token sequence** instead of one monolithic coarse path code

That would preserve:
- explicit path-shape support

while reducing:
- codebook entanglement
- all-or-nothing path classification difficulty

### Bottom line
The branch is now clearer than before:
- latent scaffolds are near cap
- explicit support is alive
- monolithic support coding is too weak

So the next principled step is to keep the explicit support idea and make it **compositional**.
