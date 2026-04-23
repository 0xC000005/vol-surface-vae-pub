## 295a-v0 postmortem

### Result
- model: `295a`
- checkpoint: `models/backfill/295a_v0_s42/best_model.pt`
- eval: `results/block_ar/295a_v0_s42/full11.json`
- score: `3/11`
- passes:
  - `surface`
  - `block_ar`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.910`
- calibration error: `0.044`
- h1 / h30 coverage90: `0.841 / 0.945`
- change KS pass: `25/25`
- level KS pass: `2/25`
- corr ratio: `1.213`
- rank ratio: `0.903`
- cointegration ratio: `0.681`
- worst-cell cointegration ratio: `0.224`
- MR ratio: `-0.005`
- active MR pass count: `0/24`
- active-cell slope corr: `0.760`
- max-jump KS: `0.620`
- jump q90 / q99 ratio: `0.870 / 1.210`
- ACF corr: `0.958`

### Training read
- best validation epoch: `1`
- best val total: `5.408`
- token NLL improved rapidly while control MAE barely moved
- later epochs overfit sharply

So the future control-state sequence was learnable enough to condition sampling, but
the predicted control states did not become a strong stable representation.

### Relative to 294a
What improved:
- coverage90: `0.782 -> 0.910`
- h1 coverage90: `0.603 -> 0.841`
- calibration error: `0.087 -> 0.044`
- cointegration ratio: `2.247 -> 0.681`
- max-jump KS: `0.662 -> 0.620`

What got worse:
- level KS: `9/25 -> 2/25`
- MR ratio: `0.163 -> -0.005`
- active MR pass count: `4/24 -> 0/24`
- worst-cell cointegration ratio: `0.211 -> 0.224` improved only slightly and still failed

What stayed wrong:
- conditionality is still weak
- regime layer-2 coverage is still below gate
- distributional fidelity still fails on level law and bias extremes

### Mechanism read
`295a` is the strongest evidence yet that the fixed-horizon joint-law family can learn
**conditional spread allocation**.

The future control-state sequence restored:
- wide and well-calibrated coverage
- short-horizon width
- non-degenerate cross-cell structure

But it did not become a usable path-shape controller.
Instead it behaved more like a volatility allocator than a mean/path allocator:
- MR collapsed entirely
- level KS gave back most of the `294a` gain
- worst-cell cointegration robustness still failed

So the family learned uncertainty better, but still failed to control the center-path
dynamics that drive the structural deterministic suites.

### Decision
Treat `295a` as the last justified in-paradigm experiment for the current fixed-horizon
one-stage joint-law line.

The next principled step is **paradigm-shift analysis**:
- do not keep iterating inside `293/294/295`
- decide whether to archive the fixed-horizon one-stage joint-law line as capped and
  move to a hybrid program that combines a stronger structural center-path mechanism
  with a learned stochastic path law
