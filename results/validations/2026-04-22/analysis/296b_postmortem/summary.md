## 296b-v0 postmortem

### Result
- model: `296b`
- checkpoint: `models/backfill/296b_v0_s42/best_model.pt`
- eval: `results/block_ar/296b_v0_s42/full11.json`
- score: `4/11`
- passes:
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`
  - `mean_reversion`

### High-signal metrics
- coverage90: `0.892`
- calibration error: `0.039`
- h1 / h30 coverage90: `0.605 / 0.956`
- change KS pass: `20/25`
- level KS pass: `0/25`
- corr ratio: `0.889`
- rank ratio: `1.536`
- cointegration ratio: `0.723`
- worst-cell cointegration ratio: `0.281`
- MR ratio: `1.088`
- active MR pass count: `21/24`
- active-cell slope corr: `0.783`
- max-jump KS: `0.446`
- ACF corr: `0.902`

### Training read
- best validation epoch: `19`
- best val total: `-0.312`
- predicted knot scales stayed non-degenerate (`~0.45`)
- normalized control magnitude stayed near unit scale (`z_abs_mean ~ 0.79`)

So unlike `296a`, the shell did not collapse into a near-deterministic corrective law.

### Relative to 296a
What improved sharply:
- score: `2/11 -> 4/11`
- coverage90: `0.483 -> 0.892`
- calibration error: `0.299 -> 0.039`
- change KS: `1/25 -> 20/25`
- cointegration ratio: `0.440 -> 0.723`
- worst-cell cointegration ratio: `0.083 -> 0.281`
- MR ratio: `1.605 -> 1.088`
- pathwise q90/q99 scale recovered

What stayed wrong:
- h1 coverage still under target
- regime differentiation is still slightly weak
- level KS is still dead
- pathwise max-jump KS remains far above gate
- surface validity missed narrowly on calendar arbitrage

### Mechanism read
`296b` validates the hybrid line much more strongly than `296a`.

The coarse, zero-mean shell did what it was supposed to do:
- preserved the backbone structural suites
- added broad stochastic width
- avoided becoming a second center path

So the main hybrid question is no longer whether the split works.
It is now narrower:
- how to allocate **more short-horizon and regime-sensitive width**
- without giving back the structural validity just recovered

### Decision
The hybrid program is still alive.

Next step:
- post-experiment analysis comparing `277d`, `296a`, and `296b`
- then specify `296c` as a short-horizon / regime-sensitive shell refinement only if
  that analysis confirms the current shell geometry is directionally right
