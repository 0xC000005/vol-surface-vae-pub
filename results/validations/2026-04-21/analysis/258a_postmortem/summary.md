# 258a-v0 Postmortem

## Result

`258a-v0` scored `3/11` on the common full 11-suite.

Passes:

- `surface`
- `block_ar`
- `cross_cell_correlation`

This ties the best stochastic family score so far (`257a`), but via a very different
mechanism.

## What Improved

Relative to the late `257` line, `258a-v0` made a real step on uncertainty and
cross-cell structure:

- coverage90: `0.343 -> 0.507`
- h30 coverage90: `0.234 -> 0.710`
- calibration error: `0.335 -> 0.238`
- cross-cell correlation suite: pass
  - `corr_ratio = 0.703`
  - `rank_ratio = 1.262`

So the family shift was not pointless. The stochastic state-space design immediately
fixed a major weakness of the token-VAE line: realistic cross-cell rank structure at
the same time as materially higher coverage.

## What Failed

The model failed badly on temporal law and jump realism:

- mean reversion ratio: `0.024`
- active-cell MR pass: `0/24`
- max-jump KS: `1.000`
- pathwise q90 ratio: `0.091`
- pathwise q99 ratio: `0.099`
- extreme jump incidence ratio: `0.000`

It also still failed:

- conditional regime widening (`turb/calm = 0.921`)
- cointegration ratio (`0.232`)
- distributional fidelity

## Mechanistic Read

The dominant mechanism is clear from training:

- `fast_scale_mean` collapsed to `~0` by epoch 2
- the fast stochastic branch effectively died
- the remaining model is mostly a slow latent state with diffuse long-horizon spread

That explains the suite shape:

- good long-horizon coverage
- good cross-cell structure
- almost no jump realism
- almost no mean-reversion strength

In other words, `258a-v0` is not yet a true slow+fast stochastic model. It is mostly
the slow branch plus a dead fast branch.

## Conclusion

`258a` is alive.

Unlike the late `257` experiments, this is not a capped-family signal. The first run
already tied the best stochastic score and fixed cross-cell rank.

The next bottleneck is very specific:

- keep the `258a` family
- prevent fast-state collapse
- make fast stochasticity survive training without destroying the improved
  cross-cell/coverage behavior

## Decision

The next principled step is **post-experiment analysis**, then a targeted `258b`
anti-collapse follow-up inside the `258` family.
