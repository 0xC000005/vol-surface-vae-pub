## 294b-v0 postmortem

### Result
- model: `294b`
- checkpoint: `models/backfill/294b_v0_s42/best_model.pt`
- eval: `results/block_ar/294b_v0_s42/full11.json`
- score: `3/11`
- passes:
  - `surface`
  - `block_ar`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.707`
- calibration error: `0.124`
- h1 / h30 coverage90: `0.000 / 0.790`
- change KS pass: `25/25`
- level KS pass: `6/25`
- corr ratio: `1.120`
- rank ratio: `1.155`
- cointegration ratio: `1.402`
- worst-cell cointegration ratio: `0.211`
- MR ratio: `0.082`
- active MR pass count: `9/24`
- active-cell slope corr: `-0.257`
- mean turbulent/calm width ratio: `1.128`
- max-jump KS: `0.740`
- jump q90 / q99 ratio: `0.752 / 1.069`
- ACF corr: `0.942`

### Training read
- best validation epoch: `3`
- best val total: `4.942`
- knot regression trained stably and with lower scaffold error than `294a`
- token accuracy remained low but stable throughout

### Relative to 294a
What improved:
- cointegration ratio: `2.247 -> 1.402`
- active MR pass count: `4/24 -> 9/24`
- worst extreme-jump scale: q99 ratio moved closer to `1.0`

What got worse:
- coverage90: `0.782 -> 0.707`
- h1 coverage90: `0.603 -> 0.000`
- calibration error: `0.087 -> 0.124`
- level KS: `9/25 -> 6/25`
- MR ratio: `0.163 -> 0.082`
- max-jump KS: `0.662 -> 0.740`

What stayed wrong:
- conditionality is still weak
- pathwise jump realism is still far below gate
- worst-cell cointegration robustness is still below gate

### Mechanism read
The piecewise-linear knot scaffold did not solve the live `294a` bottleneck.

It helped the scaffold match a more reasonable long-run cointegration level, but it
did so by weakening the short-horizon support further:
- day-1 coverage collapsed to zero
- aggregate MR weakened even more
- level-law fidelity gave back part of the `294a` gain

So the main issue is not just that the cosine scaffold was too globally smooth.
The deeper issue is that the current explicit scaffold family still does not place
enough short-horizon uncertainty and reversion mass where it needs to be.

### Decision
Do **post-experiment analysis** next.

Compare:
- `294a`
- `294b`

Question:
- is the continuous scaffold family still alive with a different scaffold-plus-residual
  formulation,
- or is the whole `294` scaffold branch already near a local cap?
