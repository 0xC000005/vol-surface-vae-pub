# 260a-v0 Postmortem

Date: 2026-04-21

## Result

- model: `260a_v0_L8_s42`
- suite score: `2/11`
- passes:
  - `block_ar`
  - `cointegration`

This is below the archived `4/11` frontier and below the best stochastic restart-adjacent score.

## High-Signal Failure Shape

`260a` is not under-dispersed.
It is the opposite.

Main evaluation fingerprints:

- overall 90% coverage: `100.0%`
- calibration error: `0.416`
- turb/calm width ratio: `0.997`
- corr ratio: `0.039`
- rank ratio: `4.060`
- h30 MR ratio: `1.615`
- max-jump KS: `0.896`
- level KS pass: `0/25`
- change KS pass: `0/25`
- floor rate: `42.8%`
- ceiling rate: `24.9%`

So the first restart baseline failed by becoming **far too broad** and then saturating against support bounds, while also losing useful common-factor structure in output space.

## Mechanism Read

The low-rank/common thesis itself did **not** fail in the expected old-family way.

Diagnostics on the trained checkpoint:

- idio/common RMS ratio: `0.239`
- idio budget mean: `0.043`
- loading effective rank mean: `8.00`
- loading top-1 share mean: `0.127`

So:

- idio is **not** dominating
- the low-rank head is active
- the model is **not** collapsing to one factor

The actual problem is scale geometry:

- generated change std: `0.719`
- GT change std: `0.109`
- change std ratio: `6.58x`
- generated level std: `0.432`
- GT level std: `0.090`
- level std ratio: `4.82x`

This means the plain FM objective on raw normalized change paths learned a common process that is far too large in magnitude. The model then reaches high coverage by brute-force widening, and the support clamp produces massive floor/ceiling occupancy.

That also explains why the cross-cell suite still fails despite an explicit low-rank head:

- the output is low-rank in construction
- but the singular spectrum is too flat and the sampled common process is too broad
- so the generated panel behaves like a wide, near-isotropic low-rank process rather than a realistic concentrated factor process

## Conclusion

`260a` cleanly falsifies the naive restart thesis:

> vanilla FM + low-rank readout in raw change space is enough

It is **not** enough.

But it fails in a useful way:

- not motif collapse
- not token collapse
- not idio leakage
- not dormant branch collapse

Instead, the restart bottleneck is now much cleaner:

**the coordinate system / target geometry is wrong for plain FM on raw normalized changes.**

## Most Principled Next Step

Stay in the restarted `260` line.

Do **not** add new architecture yet.

The next experiment should be a **coordinate/representation fix with the same architecture**, e.g.:

- causal local-scale normalized change target
- optionally asinh-transformed change target

while keeping:

- vanilla FM core
- same temporal backbone
- same low-rank readout
- same bounded idio path

That is the cleanest way to test whether the restart needs a better coordinate system before it needs a more complex architecture.
