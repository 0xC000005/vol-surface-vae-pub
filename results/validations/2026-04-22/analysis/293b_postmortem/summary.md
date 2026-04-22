## 293b-v0 postmortem

### Result
- model: `293b`
- checkpoint: `models/backfill/293b_v0_s42/best_model.pt`
- eval: `results/block_ar/293b_v0_s42/full11.json`
- score: `4/11`
- passes:
  - `surface`
  - `block_ar`
  - `cointegration`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.895`
- calibration error: `0.032`
- conditional width ratio: `1.031`
- turb/calm width ratio: `0.980`
- ACF corr: `0.956`
- change KS pass: `23/25`
- level KS pass: `2/25`
- corr ratio: `1.256`
- rank ratio: `0.879`
- cointegration ratio: `0.702`
- MR ratio: `0.034`
- max-jump KS: `0.680`

### Relative to 293a
What improved:
- score: `3/11 -> 4/11`
- calibration error: `0.041 -> 0.032`
- cointegration now passes
- worst-cell cointegration ratio recovered above gate

What did not improve:
- level KS stayed at `2/25`
- regime differentiation remained dead
- MR got worse (`0.082 -> 0.034`)
- pathwise max-jump KS only improved marginally (`0.700 -> 0.680`)

### Mechanism read
The global path latent did **not** become the missing path-structure mechanism.

What it appears to have done:
- provide a slightly better window-level dispersion / dependence prior
- help calibration and cointegration a bit

What it did **not** do:
- enforce the directional long-horizon dynamics that matter for:
  - mean reversion
  - level-law fidelity
  - regime-sensitive width allocation
  - pathwise jump ordering

So the clean read is:
- `293b` buys a small global-window correlation benefit
- but the current latent is acting more like an extra entropy/context channel than a true long-horizon path-shape controller

### Family status
`293` remains alive, but this specific `293b` mechanism is not the breakthrough.

The family now has evidence for:
- live stochastic spread
- strong daily change-law fidelity
- usable cross-cell structure
- recoverable cointegration

The remaining blocker is narrower:
- **path-shape control**, not just path-level entropy

### Most principled next step
Do **post-experiment analysis** next.

Specifically compare:
- `293a`
- `293b`

Question:
- what moved with the global latent, and what stayed invariant?

The likely next move should target **path-shape coupling**, not another generic latent-capacity increase.
