## 293g-v0 postmortem

### Result
- model: `293g`
- checkpoint: `models/backfill/293g_v0_s42/best_model.pt`
- eval: `results/block_ar/293g_v0_s42/full11.json`
- score: `4/11`
- passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.948`
- calibration error: `0.109`
- change KS pass: `14/25`
- level KS pass: `3/25`
- corr ratio: `0.690`
- rank ratio: `1.933`
- cointegration ratio: `1.035`
- worst-cell cointegration ratio: `0.250`
- MR ratio: `2.420`
- active-cell pass rate: `12.5%`
- active-cell corr: `0.703`
- turb/calm width ratio: `0.697`
- max-jump KS: `0.578`
- jump q90 ratio: `1.186`
- jump q99 ratio: `1.232`

### Training read
- best validation epoch: `6`
- best val total: `6.480`
- residual token accuracy was materially stronger than the earlier support runs
- coarse support accuracy also came alive relative to the earlier anchor branches

So the residual formulation was easier to optimize than the previous support-object variants.

### Relative to 293f
What improved:
- coverage90: `0.913 -> 0.948`
- level KS: `2/25 -> 3/25`
- cointegration ratio: `0.779 -> 1.035`
- max-jump q90/q99 scale came into gate
- active-cell MR corr: `0.591 -> 0.703`

What regressed:
- change KS: `25/25 -> 14/25`
- calibration error: `0.054 -> 0.109`
- corr ratio: `1.278 -> 0.690`
- MR ratio overshot badly: `0.098 -> 2.420`
- regime differentiation inverted: `1.049 -> 0.697`
- max-jump KS stayed far from gate: `0.660 -> 0.578`

### Mechanism read
`293g` is important because it changed the behavior in the predicted direction:
- the scaffold was no longer being ignored
- support use became structurally active

But the first residual formulation was too aggressive.

Using the scaffold as a daily **level pull target** caused:
- over-mean-reversion
- regime-width inversion
- too much long-horizon coverage inflation
- loss of daily change-law sharpness

So the result is not "support use is wrong."
It is:
- **support use is live**
- but the current residual formulation over-anchors to the scaffold

### Family status
`293` is still alive.

The support branch is no longer a support-representation problem.
It is now a support-use formulation problem.

### Most principled next step
Do **research ideation** next.

The clean next question is:
- can the scaffold act as a **drift schedule** rather than a direct daily level attractor?

That means:
- baseline on scaffold increments or low-frequency drift
- residual law around that schedule
- not another support-object change
