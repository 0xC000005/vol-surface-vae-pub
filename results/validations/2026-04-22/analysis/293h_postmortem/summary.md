## 293h-v0 postmortem

### Result
- model: `293h`
- checkpoint: `models/backfill/293h_v0_s42/best_model.pt`
- eval: `results/block_ar/293h_v0_s42/full11.json`
- score: `4/11`
- passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.980`
- calibration error: `0.143`
- change KS pass: `25/25`
- level KS pass: `0/25`
- corr ratio: `1.027`
- rank ratio: `1.234`
- cointegration ratio: `0.811`
- worst-cell cointegration ratio: `0.263`
- MR ratio: `2.321`
- active cells: `1/24`
- active-cell corr: `0.632`
- turb/calm width ratio: `0.876`
- max-jump KS: `0.643`
- jump q90 ratio: `0.816`
- jump q99 ratio: `1.163`

### Training read
- best validation epoch: `4`
- best val total: `8.644`
- residual token accuracy stayed very low throughout training
- coarse support accuracy was modest but real

So the increment-geometry variant was much harder to fit than `293g`.

### Relative to 293g
What improved:
- change KS: `14/25 -> 25/25`
- corr ratio: `0.690 -> 1.027`
- rank ratio: `1.933 -> 1.234`
- worst-cell width pathologies eased somewhat
- floor/ceiling behavior stayed within gate

What got worse:
- level KS: `3/25 -> 0/25`
- coverage90: `0.948 -> 0.980`
- calibration error: `0.109 -> 0.143`
- active-cell MR corr: `0.703 -> 0.632`
- max-jump KS: `0.578 -> 0.643`

What stayed structurally wrong:
- aggregate MR ratio remained badly over-reverted
- regime width timing stayed below gate
- distributional fidelity still failed on level-law and bias gates

### Mechanism read
`293h` confirms the support-use diagnosis more sharply:
- changing the scaffold geometry really does move the behavior
- the branch is not frozen

But the drift-schedule baseline still does not solve the core path-shape gap.

Compared with `293g`, it restores the local change-law and cross-cell structure that the direct level-pull baseline damaged.
But it still leaves:
- level-law fidelity dead
- mean reversion too strong
- regime differentiation too weak

So the support-use subfamily is now looking close to a local cap too.

### Most principled next step
Do **post-experiment analysis** next.

Specifically compare:
- `293f`
- `293g`
- `293h`

Question:
- did the support-use variants actually open a live path,
- or did they just expose a clean but capped tradeoff between:
  - respecting the scaffold strongly enough to matter
  - and preserving the local stochastic law?
