## 296d-v0 postmortem

### Result
- model: `296d`
- checkpoint: `models/backfill/296d_v0_s42/best_model.pt`
- eval: `results/block_ar/296d_v0_s42/full11.json`
- score: `4/11`
- passes:
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`
  - `mean_reversion`

### High-signal metrics
- coverage90: `0.890`
- calibration error: `0.024`
- h1 / h30 coverage90: `0.576 / 0.967`
- change KS pass: `19/25`
- level KS pass: `0/25`
- corr ratio: `0.849`
- rank ratio: `1.678`
- cointegration ratio: `0.745`
- worst-cell cointegration ratio: `0.281`
- MR ratio: `1.092`
- active MR pass count: `21/24`
- active-cell slope corr: `0.780`
- max-jump KS: `0.439`
- q99 ratio: `0.928`

### Training read
- best validation epoch: `14`
- best val total: `0.624`
- learned tail heaviness stayed active instead of collapsing back to Gaussian:
  - best-epoch val `nu_mean ~ 5.66`
  - best-epoch val `nu_min ~ 3.82`
- scale predictions stayed non-degenerate (`pred_scale_mean ~ 0.39`)

So the support change was real. The shell learned a mildly heavy-tailed residual law.

### Relative to 296c
What improved:
- overall coverage90: `0.882 -> 0.890`
- calibration error: `0.031 -> 0.024`
- MAE reduction: `-0.5% -> 2.1%`
- cointegration ratio: `0.744 -> 0.745`
- max-jump KS: `0.453 -> 0.439`
- q99 ratio: `0.900 -> 0.928`

What got worse:
- score: `5/11 -> 4/11`
- surface validity lost narrowly on calendar arbitrage
- h1 coverage90: `0.593 -> 0.576`
- turb/calm width ratio: `1.110 -> 1.049`
- corr ratio: `0.904 -> 0.849`
- rank ratio: `1.491 -> 1.678`
- floor rate: `2.11% -> 2.13%`

### Mechanism read
The Student-t shell support changed the **tail law**, but it did not solve the real
allocation problem.

It helped:
- broad calibration
- pathwise tail scale
- slight MAE conditionality

But it hurt or failed to improve:
- short-horizon width where the suite is tightest
- regime-sensitive width allocation
- structural carryover from the backbone

So the remaining bottleneck is not "make the shell heavier-tailed everywhere".
It is "allocate shell mass in the right regimes and horizons without giving back
structure".

### Decision
Do post-experiment analysis comparing `296c` and `296d`.

Key question:
- did `296d` prove the support-law branch is alive but underdirected,
- or did it show that simple support changes are already near a local cap inside the
  hybrid line?
