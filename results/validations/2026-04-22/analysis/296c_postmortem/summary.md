## 296c-v0 postmortem

### Result
- model: `296c`
- checkpoint: `models/backfill/296c_v0_s42/best_model.pt`
- eval: `results/block_ar/296c_v0_s42/full11.json`
- score: `5/11`
- passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`
  - `mean_reversion`

### High-signal metrics
- coverage90: `0.882`
- calibration error: `0.031`
- h1 / h30 coverage90: `0.593 / 0.953`
- change KS pass: `19/25`
- level KS pass: `0/25`
- corr ratio: `0.904`
- rank ratio: `1.491`
- cointegration ratio: `0.744`
- worst-cell cointegration ratio: `0.281`
- MR ratio: `1.086`
- active MR pass count: `21/24`
- active-cell slope corr: `0.778`
- max-jump KS: `0.453`
- ACF corr: `0.906`

### Training read
- best validation epoch: `22`
- best val total: `-0.303`
- query-conditioned profile scale stabilized around `0.54`
- h1 profile component stabilized around `0.76`
- no shell-collapse behavior appeared

### Relative to 296b
What improved:
- score: `4/11 -> 5/11`
- surface validity now passes
  - calendar arbitrage: `15.5% -> 14.9%`
- calibration error: `0.039 -> 0.031`
- cointegration ratio: `0.723 -> 0.744`
- corr ratio: `0.889 -> 0.904`

What did not improve:
- h1 coverage: `0.605 -> 0.593`
- turb/calm width ratio: `1.141 -> 1.110`
- level KS: still `0/25`
- pathwise max-jump KS: `0.446 -> 0.453`

### Mechanism read
`296c` confirms that the hybrid line is genuinely competitive again.

The shared query-conditioned knot profile helped with:
- support/surface behavior
- preserving the structural backbone
- slight global calibration cleanup

But it did **not** solve the actual remaining bottleneck:
- short-horizon width is still underallocated
- regime-sensitive width is still too weak
- pathwise jump timing is still too smooth

So the simple scale-profile factorization is directionally acceptable, but likely
insufficient by itself.

### Decision
The hybrid line stays alive and now matches the best overall frontier at `5/11`.

Next step:
- post-experiment analysis comparing `296b` and `296c`
- decide whether the scale-allocation subfamily is near a local cap
- if so, `296d` should change shell **support / objective**, not just another scale head
