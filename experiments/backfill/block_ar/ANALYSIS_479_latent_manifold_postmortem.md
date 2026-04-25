# 479 Latent-Manifold Branch Postmortem

## Context

The 475 paradigm shift tested whether a learned future-path bottleneck could make
the conditional 30-day level law easier than either:

- direct raw future-path modeling, which repeatedly lost stochastic geometry; or
- wrappers around `392a`, which preserved geometry but could not learn
  conditional level/regime allocation.

The branch stayed clean:

- deterministic empirical-normal-score future-path autoencoder;
- vanilla conditional latent rectified flow;
- no `392a` wrapper, retrieval, calibration table, low-rank readout, bounded side
  path, regime label, or validation-future oracle.

## Results

| run | diagnostic | key result |
| --- | --- | --- |
| `476a` | full conditional latent flow | `4/11`; passes surface, block-AR, cointegration, cross-cell correlation |
| `477a` | 476a reconstruction oracle | daily KS `20/25`, level KS `9/25`, MR ratio `1.53`, pathwise KS `0.969` |
| `478a` | larger 256-latent reconstruction oracle | daily KS `21/25`, level KS `10/25`, MR ratio `1.59`, pathwise KS `0.875` |

Important comparison:

- `476a` conditional sampling failed conditionality (`-4.23%` MAE reduction),
  level KS (`2/25`), mean reversion, and pathwise jumps.
- The reconstruction oracle showed that this was not only a conditional-flow
  problem: even when the autoencoder sees the true future path, it cannot preserve
  the validation level and first-step path geometry.
- Increasing latent capacity from `96` to `256` improved local change scale and
  cross-cell correlation, but level occupancy, median-bias, mean-reversion, and
  pathwise max-jump geometry remained failed.

## Mechanism

The learned bottleneck is not degenerate. It can learn a path manifold with:

- realistic broad time-series shape;
- good cross-cell correlation and effective rank;
- improved daily-change KS under oracle reconstruction.

But it acts like a smoothing projection on the validation future paths:

- reconstructed score variance remains below target;
- daily-change standard deviation remains compressed;
- level KS stays far below the `15/25` gate;
- first-step mean-reversion slopes become too strong and too homogeneous;
- pathwise max-jump distribution remains too different.

This is the same broad tradeoff seen in older bottleneck lines: compression helps
shared geometry, but the hard suite depends on localized level occupancy and jump
geometry that the deterministic bottleneck does not retain.

## Decision

Close the deterministic latent future-path autoencoder as a primary route.

Do not continue with:

- latent-dimension sweeps;
- decoder-width sweeps;
- latent-flow temperature tuning;
- calibration layers on top of the latent generator.

The active deployable frontier remains `392a` at `8/11`.

## Next Direction

The next paradigm should preserve two lessons:

1. `392a` proves that local AR transition geometry is learnable and deployable.
2. The latent branch proves that compressing the full future path before modeling
   it loses validation level/jump details.

Therefore the next candidate should not be a smoothed full-path bottleneck. It
should model the future law through a mathematically exact decomposition that
keeps local path evolution explicit while giving long-horizon level occupancy a
native random variable. The cleanest candidate to ideate next is an endpoint /
bridge factorization:

```text
p(Y_1:T | H) = p(Y_T or coarse knots | H) * p(Y_1:T-1 | H, endpoint/coarse knots)
```

This is not a calibration shell or low-rank assumption; it is an exact probability
decomposition. The risk is that older coarse-knot experiments already showed
similar anchor/transition tradeoffs, so the next step must be ideation/postmortem
before implementation, not an immediate bridge model.
