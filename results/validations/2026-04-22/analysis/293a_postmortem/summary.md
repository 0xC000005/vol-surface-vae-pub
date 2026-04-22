## 293a-v0 postmortem

### Result
- model: `293a`
- checkpoint: `models/backfill/293a_v0_s42/best_model.pt`
- eval: `results/block_ar/293a_v0_s42/full11.json`
- score: `3/11`
- passes:
  - `surface`
  - `block_ar`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.901`
- calibration error: `0.041`
- conditional width ratio: `1.014`
- turb/calm width ratio: `1.010`
- ACF corr: `0.957`
- kurtosis ratio: `1.735`
- change KS pass: `24/25`
- level KS pass: `2/25`
- corr ratio: `1.245`
- rank ratio: `0.880`
- cointegration ratio: `0.705`
- MR ratio: `0.082`
- active MR pass rate: `0/24`
- max-jump KS: `0.700`
- jump q90 / q99: `0.821 / 1.220`

### Mechanism read
`293a-v0` is a real family result, not a dead baseline.

What it learned correctly:
- stochastic spread is alive immediately
- broad calibration is alive immediately
- the daily change law is much better than the deterministic world-model line
- cross-cell dependence is preserved rather than collapsing into per-cell factorization

What it still gets wrong:
- cumulative level law is poor (`level KS 2/25`)
- mean reversion is nearly absent (`MR ratio 0.082`)
- regime differentiation is too weak (`turb/calm 1.010`)
- pathwise jump ordering / incidence over the full window is still wrong (`max-jump KS 0.700`)

The clean causal read is:
- the fixed-horizon **joint token likelihood** is enough to learn a useful local conditional move law
- but the current day-token chain does not impose enough **global path structure** across the 30-day window
- so the model matches daily change marginals and broad cross-cell structure while missing the long-horizon path constraints that drive:
  - levels
  - MR
  - regime-sensitive width allocation
  - pathwise jump realism

In short:
- local law: alive
- path law: weak

### Family status
`293a` stays alive.

This run falsifies neither:
- fixed-horizon conditional joint-law modeling
- nor discrete joint support as a minimal `v0`

It does falsify the idea that:
- a plain daily joint-token chain is already enough for the 30-day path law

### Most principled next step
Do **research ideation**, not another blind training tweak.

The next mechanism should preserve:
- the live stochastic spread
- the strong change KS
- the usable cross-cell structure

while adding exactly one missing ingredient:
- a **global path-level coupling mechanism** across the full future window

Candidate direction for `293b`:
- keep the fixed-horizon joint-law family
- keep the daily joint support primitive
- add one global future latent / path summary that conditions all daily token decodes

That is the narrowest move that directly targets the observed failure:
- weak long-horizon path structure under an otherwise live local joint law
