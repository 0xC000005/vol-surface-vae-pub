# 431a: Joint Future Likelihood Flow

## Context

The active learned frontier remains `392a` at `8/11`.

The oracle-calibration branch is closed as a primary route:

- per-horizon/cell marginal maps can repair level occupancy but break conditional
  structure;
- path-center shifts preserve more structure but fail coverage/regime geometry;
- post-hoc correction is therefore the wrong object for a publishable model.

The next model must learn the joint 30-day future law directly.

## Hypothesis

An exact-likelihood conditional normalizing flow over the full future path may model
level occupancy natively while avoiding the sampler-temperature/objective mismatch of
the failed direct path-flow-matching branch.

This is a single-stage conditional density model:

1. Encode the 30-day history.
2. Transform the full 30-day future path into empirical normal-score coordinates.
3. Fit an invertible conditional coupling flow with exact log likelihood.
4. Sample the joint future path in one shot.

## Minimal Architecture

Use one RealNVP-style conditional affine coupling flow over the flattened future tensor:

- future dimension: `30 * 5 * 5 = 750`;
- base distribution: standard Gaussian;
- data transform: empirical normal-score CDF shared across train futures/histories;
- conditioner: compact GRU or Transformer history encoder;
- coupling layers: alternating fixed binary masks over flattened future variables;
- coupling network: MLP taking masked future plus history context, outputting scale/shift
  for unmasked variables;
- scale clamp: bounded `tanh` scale only for numerical invertibility, not a bounded idio
  path or model prior.

No posterior/prior scaffold, no low-rank decoder, no bounded EC/idiosyncratic path, and
no post-hoc evaluator correction.

## Why This Is Different From Failed Branches

This differs from the direct future path-FM variants (`413a`/`417a`) because the training
objective is exact conditional likelihood, not velocity matching plus ODE/sampler
temperature choices.

It differs from the AR transition family because the 30-day level law is not an emergent
property of recursive one-day transitions. The model assigns likelihood to the whole
future path jointly.

It differs from calibration/oracle systems because no validation-future output transform
is applied after sampling.

## Falsifier

Implement one minimal model and evaluate unchanged on the full 11-suite.

Success:

- beats `392a` at `8/11`, or
- cleanly combines native level occupancy with at least most of `392a`'s structural
  passes.

Failure:

- repeats direct-path failures: weak conditionality, broken cross-cell geometry, or poor
  mean reversion;
- learns marginal levels but fails structural suites;
- collapses to overbroad unconditional sampling.

If it fails in that pattern, close exact-likelihood joint path flow and revisit the
product/test target rather than adding coupling-depth or mask sweeps.

