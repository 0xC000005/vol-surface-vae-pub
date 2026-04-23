## 296f-v0 postmortem

### Result
- model: `296f`
- checkpoint: `models/backfill/296f_v0_s42/best_model.pt`
- eval: `results/block_ar/296f_v0_s42/full11.json`
- score: `4/11`
- passes:
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`
  - `mean_reversion`

### High-signal metrics
- coverage90: `0.928`
- calibration error: `0.076`
- h1 / h30 coverage90: `0.862 / 0.956`
- change KS pass: `17/25`
- level KS pass: `0/25`
- corr ratio: `0.772`
- rank ratio: `1.912`
- cointegration ratio: `0.739`
- worst-cell cointegration ratio: `0.263`
- MR ratio: `1.113`
- active-cell slope corr: `0.800`
- turb/calm width ratio: `1.117`
- max-jump KS: `0.389`
- q99 ratio: `1.188`

### Training read
- best validation epoch: `16`
- best val total: `-0.029`
- the budget factorization stayed active:
  - best-epoch `pred_scale_daily_mean ~ 0.483`
  - best-epoch `budget_mean ~ 0.483`
- so the shell was actually respecting the learned average budget instead of using
  the fast basis to inflate width freely

### Relative to 296e
What improved:
- overall coverage90: `0.930 -> 0.928`
- calibration error: `0.082 -> 0.076`
- MAE reduction: `-1.0% -> 0.5%`
- change KS: `16/25 -> 17/25`
- corr ratio: `0.750 -> 0.772`
- rank ratio: `2.006 -> 1.912`
- cointegration ratio: `0.734 -> 0.739`
- max-jump KS: `0.394 -> 0.389`

What stayed wrong:
- score stayed `4/11`
- h1 coverage stayed high but still overexpressed globally
- turb/calm width ratio fell back below gate: `1.165 -> 1.117`
- surface validity still failed
- level KS stayed dead

### Mechanism read
`296f` partially validated the budgeted multiresolution idea.

It did the intended thing:
- kept the fast-shell local gains from `296e`
- pulled the model slightly back toward `296c` on calibration and structure

But the correction was too weak:
- enough to reduce the damage
- not enough to restore a balanced shell

So the next bottleneck is now even narrower:
- not whether the shell needs a budget
- but how to make the fast-shell budget **regime-aware / locally activated** instead
  of always present

### Decision
Do post-experiment analysis comparing `296e` and `296f`.

Key question:
- does `296f` show that constrained multiresolution shell is still the right family,
- and if so, is the next missing ingredient a gating/activation mechanism rather than
  another global budget tweak?
