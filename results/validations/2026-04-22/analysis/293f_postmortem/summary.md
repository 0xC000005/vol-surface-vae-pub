## 293f-v0 postmortem

### Result
- model: `293f`
- checkpoint: `models/backfill/293f_v0_s42/best_model.pt`
- eval: `results/block_ar/293f_v0_s42/full11.json`
- score: `4/11`
- passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.913`
- calibration error: `0.054`
- change KS pass: `25/25`
- level KS pass: `2/25`
- corr ratio: `1.278`
- rank ratio: `0.863`
- cointegration ratio: `0.779`
- worst-cell cointegration ratio: `0.250`
- MR ratio: `0.098`
- active-cell corr: `0.591`
- turb/calm width ratio: `1.049`
- max-jump KS: `0.660`

### Training read
- best validation epoch: `4`
- best val total: `9.042`
- the hybrid stayed numerically stable
- coarse anchor accuracy remained weak
- refinement loss stayed bounded and stable

So the anchor-refine hybrid trained stably, but the anchor head still did not become strongly predictive.

### Relative to 293d
What improved:
- coverage90: `0.909 -> 0.913`
- cointegration ratio: `0.656 -> 0.779`
- worst-cell cointegration ratio recovered to the exact gate
- MR ratio: `0.093 -> 0.098`
- turb/calm ratio: `0.984 -> 1.049`

What stayed broadly similar:
- score remained `4/11`
- change KS stayed `25/25`
- cross-cell structure stayed in gate
- level KS remained effectively dead: `1/25 -> 2/25`

What regressed:
- active-cell MR corr: `0.773 -> 0.591`
- jump KS: `0.647 -> 0.660`
- calibration error: `0.049 -> 0.054`

### Relative to 293e
What improved:
- score: `3/11 -> 4/11`
- change KS: `24/25 -> 25/25`
- cointegration robustness recovered
- MR ratio increased

What did not keep:
- `293e`'s better level KS (`3/25`)
- `293e`'s better jump KS (`0.625`)

### Mechanism read
`293f` did not break the support-object tradeoff.

It kept more of the structural anchoring side:
- change KS
- cointegration
- some MR support

But it did not keep enough of the soft path-shape gains:
- level KS barely moved
- jump realism regressed versus `293e`
- active MR correlation regressed versus `293d`

So the hybrid mostly behaved like:
- a slightly better anchored variant
- not a synthesis of both sides

### Family status
`293` is still alive at the local-law level.

But the explicit support-object branch now looks close to a local cap:
- `293d`: anchor-heavy
- `293e`: refinement-heavy
- `293f`: hybrid, but still not a breakout

### Most principled next step
Do **post-experiment analysis** next.

Specifically compare:
- `293d`
- `293e`
- `293f`

Question:
- did the hybrid actually resolve the tradeoff,
- or did it just land closer to the `293d` side while leaving the core path-shape suites dead?
