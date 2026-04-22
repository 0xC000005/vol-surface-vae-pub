## 293e-v0 postmortem

### Result
- model: `293e`
- checkpoint: `models/backfill/293e_v0_s42/best_model.pt`
- eval: `results/block_ar/293e_v0_s42/full11.json`
- score: `3/11`
- passes:
  - `surface`
  - `block_ar`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.901`
- calibration error: `0.038`
- change KS pass: `24/25`
- level KS pass: `3/25`
- corr ratio: `1.215`
- rank ratio: `0.914`
- cointegration ratio: `0.768`
- MR ratio: `0.075`
- active-cell corr: `0.670`
- turb/calm width ratio: `1.029`
- max-jump KS: `0.625`

### Training read
- best validation epoch: `2`
- best val total: `8.772`
- coarse support-token accuracy stayed modest but steadier than `293d`
  - around `0.14 - 0.17` on validation in the early and mid run

So the compositional support representation was somewhat easier to learn than the monolithic coarse code.

### Relative to 293d
What improved:
- calibration error: `0.049 -> 0.038`
- level KS: `1/25 -> 3/25`
- rank ratio: `0.849 -> 0.914`
- max-jump KS: `0.647 -> 0.625`
- turb/calm width ratio: `0.984 -> 1.029`

What regressed:
- score: `4/11 -> 3/11`
- change KS: `25/25 -> 24/25`
- cointegration ratio: `0.656 -> 0.768`, but worst-cell ratio fell below gate again
- aggregate MR ratio: `0.093 -> 0.075`
- active-cell MR corr: `0.773 -> 0.670`

### Mechanism read
The compositional support idea is not a no-op.

It improved some of the exact suites that a path-shape mechanism should touch:
- level KS
- regime width differentiation
- pathwise jump KS

But it simultaneously weakened the structural gains that made `293d` promising:
- cointegration robustness
- MR structure
- perfect daily change KS

So the direct read is:
- `293e` made the support representation easier to predict
- but it also made the coarse scaffold less structurally anchoring

That means the branch is now in a genuine support-object tradeoff:
- `293d`: stronger structural anchor, weaker predictability
- `293e`: easier support prediction, weaker structural anchor

### Family status
`293` is still alive.

But the support-object subfamily is no longer monotonic:
- monolithic support recovers more structure
- compositional support recovers more local/path-shape softness

So the next move should not be another blind support variant.

### Most principled next step
Do **post-experiment analysis** next.

Specifically compare:
- `293d`
- `293e`

Question:
- is there a clean way to preserve the structural anchoring of `293d` while recovering the easier learnability of `293e`,
- or does the support-object family itself now need a harder rethink?
