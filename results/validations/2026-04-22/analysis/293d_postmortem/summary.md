## 293d-v0 postmortem

### Result
- model: `293d`
- checkpoint: `models/backfill/293d_v0_s42/best_model.pt`
- eval: `results/block_ar/293d_v0_s42/full11.json`
- score: `4/11`
- passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.909`
- calibration error: `0.049`
- change KS pass: `25/25`
- level KS pass: `1/25`
- corr ratio: `1.301`
- rank ratio: `0.849`
- cointegration ratio: `0.656`
- MR ratio: `0.093`
- active-cell corr: `0.773`
- turb/calm width ratio: `0.984`
- max-jump KS: `0.647`

### Training read
- best validation epoch: `3`
- best val total: `8.812`
- coarse-code accuracy stayed low:
  - best early values around `0.10-0.15`
  - later values remained below `0.10`

So the explicit coarse support object was hard to predict, but the model still extracted some usable signal from it.

### Relative to 293c
What improved:
- score: `3/11 -> 4/11`
- change KS: `23/25 -> 25/25`
- cointegration ratio: `0.610 -> 0.656`
- aggregate MR ratio: `0.057 -> 0.093`
- active-cell MR corr: `0.523 -> 0.773`
- worst-cell cointegration ratio recovered into pass

What stayed dead:
- level KS: `1/25`
- regime differentiation: `0.990 -> 0.984`
- pathwise max-jump KS still far from gate: `0.674 -> 0.647`

What got worse:
- calibration error: `0.023 -> 0.049`

### Mechanism read
The explicit coarse path support object is directionally better than the latent scaffold.

Why:
- it restored cointegration pass
- it recovered some mean-reversion structure
- it preserved the family's strong local-law behavior

That is a different pattern from `293c`.

But it is still not enough:
- level-law fidelity remains essentially absent
- regime-sensitive width timing remains dead
- jump ordering is still far from gate

So the coarse support idea is **alive**, but the first implementation is too weak. The support object helped as a structural anchor, but the current coarse-code prediction head is not strong enough to make it a decisive controller of path shape.

### Family status
`293` remains alive.

The branch status is now:
- latent-conditioning variants: near local cap
- explicit coarse-support variant: directionally alive, but still insufficient

### Most principled next step
Do **post-experiment analysis** next.

Specifically compare:
- `293b`
- `293c`
- `293d`

Question:
- what did the explicit support object recover that hidden latents could not,
- and is the next move to strengthen the support-object prediction / structure rather than change the local joint-law core again?
