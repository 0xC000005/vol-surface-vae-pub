## 296a-v0 postmortem

### Result
- model: `296a`
- checkpoint: `models/backfill/296a_v0_s42/best_model.pt`
- eval: `results/block_ar/296a_v0_s42/full11.json`
- score: `2/11`
- passes:
  - `block_ar`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.483`
- calibration error: `0.299`
- h1 / h30 coverage90: `0.182 / 0.531`
- change KS pass: `1/25`
- level KS pass: `0/25`
- corr ratio: `0.900`
- rank ratio: `1.291`
- cointegration ratio: `0.440`
- worst-cell cointegration ratio: `0.083`
- MR ratio: `1.605`
- active MR pass count: `19/24`
- active-cell slope corr: `0.748`
- max-jump KS: `0.723`
- ACF corr: `0.924`

### Training read
- best validation epoch: `24`
- best val total: `0.143`
- shell token accuracy reached `~97%`
- shell token entropy collapsed to `~0.13`
- residual raw MAE plateaued near `0.0395`

So the shell fit was numerically easy and highly deterministic.

### Relative to the intended hybrid goal
`296a` was supposed to combine:
- `277d` structural center-path strength
- `295a` stochastic spread allocation

It did neither.

What stayed alive:
- boundary smoothness
- cross-cell mean/rank structure
- broad time-series ACF

What failed immediately:
- coverage collapsed
- calibration collapsed
- cointegration dropped below gate
- surface validity failed again
- level-law fidelity stayed dead
- jump realism stayed poor

### Mechanism read
The hybrid decomposition itself is **not yet validated** by this implementation.

The failure is specific:
1. The daily residual shell objective collapsed toward a near-deterministic corrective
   shell.
2. That shell did not preserve the fixed `277d` center path under rollout.
3. Because residuals were trained as direct daily token corrections, the shell
   introduced strong pathwise bias and over-reversion while still remaining too narrow.

So the current interface is wrong:
- too local
- too deterministic under cross-entropy training
- too able to drag the structural backbone away from its own valid center path

### Decision
Do **not** continue directly to another hybrid shell experiment.

The next principled step is **post-experiment analysis**:
- compare `277d`, `295a`, and `296a`
- isolate whether the failure comes primarily from:
  - the daily residual representation, or
  - the shell training objective collapsing to a narrow corrective median

Only after that should `296b` be specified.
