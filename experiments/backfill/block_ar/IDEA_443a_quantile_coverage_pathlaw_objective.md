# 443a: Quantile-Coverage Path-Law Objective

## Why Another Learned Objective

`410a` already tried marginal CRPS and did not beat `392a`. The next objective must not
be a duplicate of CRPS under a different name.

The remaining failures point to two specific learned-law defects:

- level KS remains below the `15/25` gate;
- coverage/regime layer2 remains miscalibrated even when overall coverage is near
  reasonable.

Post-hoc calibration and residual banks do not fix this without damaging conditionality
or time-series realism. The model itself needs to learn a better conditional path-law.

## Objective

Start from `392a` and fine-tune on pre-validation recent windows with four terms:

1. FM anchor: preserve one-step conditional transition learning.
2. Path-energy anchor: preserve the `392a` free-running path geometry.
3. IV-level quantile matching: match generated and realized unconditional level
   quantiles across batch, horizon, and cell.
4. Interval pinball loss: make generated q05/q95 behave like calibrated predictive
   quantile boundaries.

This differs from `410a` because it targets explicit level quantiles and interval
boundaries in IV space, not only mean univariate CRPS in score space.

## Cleanliness

This is still a learned conditional generator:

- no validation futures;
- no post-hoc residual bank;
- no evaluator-specific correction;
- no architecture change;
- one loss-family change with interpretable scoring-rule motivation.

## Falsifier

Run `444a` from `392a` with modest quantile/coverage weights.

Success means improving the `392a` frontier, especially level KS and coverage/regime
layer2, without sacrificing conditionality/time-series/structure.

Failure means this loss family is likely not enough, and the next move should be a
larger learned-core change rather than more fine-tune objective variants.
