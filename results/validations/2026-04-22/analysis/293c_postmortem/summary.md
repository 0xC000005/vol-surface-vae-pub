## 293c-v0 postmortem

### Result
- model: `293c`
- checkpoint: `models/backfill/293c_v0_s42/best_model.pt`
- eval: `results/block_ar/293c_v0_s42/full11.json`
- score: `3/11`
- passes:
  - `surface`
  - `block_ar`
  - `cross_cell_correlation`

### High-signal metrics
- coverage90: `0.892`
- calibration error: `0.023`
- change KS pass: `23/25`
- level KS pass: `1/25`
- corr ratio: `1.244`
- rank ratio: `0.881`
- cointegration ratio: `0.610`
- MR ratio: `0.057`
- turb/calm width ratio: `0.990`
- ACF corr: `0.960`
- max-jump KS: `0.674`

### Training read
- best validation epoch: `1`
- best val total: `5.243`
- by the final epoch:
  - val total deteriorated to `9.443`
  - val token entropy fell from `4.783 -> 1.530`
  - val token accuracy improved only marginally

So the run stayed numerically stable, but overfit almost immediately.

### Relative to 293b
What improved:
- calibration error: `0.032 -> 0.023`
- max-jump KS: `0.680 -> 0.674` (still far from gate)
- aggregate MR ratio: `0.034 -> 0.057` (still effectively dead)

What stayed unchanged:
- coverage stayed alive
- change KS stayed strong: `23/25`
- cross-cell structure stayed in gate:
  - corr ratio: `1.256 -> 1.244`
  - rank ratio: `0.879 -> 0.881`

What got worse:
- score: `4/11 -> 3/11`
- level KS: `2/25 -> 1/25`
- cointegration ratio: `0.702 -> 0.610`
- regime differentiation stayed dead: `0.980 -> 0.990`

### Mechanism read
The time-structured knot latent did **not** become the missing path-shape controller.

It preserved the same family invariants as `293a/293b`:
- live local stochastic move law
- good calibration
- usable cross-cell structure

But it still failed on the actual path-shape suites:
- level-law fidelity
- mean reversion
- regime width timing
- jump ordering

The cleanest read is:
- replacing one global latent with interpolated knot latents is **not enough by itself**
- the decoder still treats the latent scaffold as extra context, not as a hard low-frequency trajectory constraint

The training dynamics support that:
- early best epoch
- rapid entropy collapse
- no commensurate gain in level-law or MR metrics

So `293c` did not falsify the whole `293` family, but it did falsify the idea that **latent time structure alone** is the next missing mechanism.

### Family status
`293` is still alive at the local-law level, but the current latent-conditioning subfamily now looks close to capped:
- `293a`: local-law alive
- `293b`: global latent helps calibration / cointegration slightly
- `293c`: time-structured latent still does not fix path shape

### Most principled next step
Do **post-experiment analysis** next.

Specifically compare:
- `293a`
- `293b`
- `293c`

Question:
- is the family missing a more explicit coarse path representation,
- or is the current tokenized local-law formulation itself structurally unable to control long-horizon path shape?
