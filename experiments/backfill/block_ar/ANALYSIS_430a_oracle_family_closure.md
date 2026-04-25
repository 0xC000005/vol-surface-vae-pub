# 430a: Oracle Family Closure

## Context

The oracle-calibration branch tested three distinct ways to repair the `392a` frontier
without changing the learned generator:

- `426a`: nonlinear per-horizon/cell marginal quantile map;
- `427a`: affine per-horizon/cell median-shift plus interval-scale map;
- `429a`: per-window path-constant center shift.

This was an explicit feasibility study, not a deployable-model claim.

## Evidence

| System | Score | What It Fixed | What It Broke / Missed |
| --- | ---: | --- | --- |
| `392a` raw | `8/11` | structural suites | coverage, regime, level distribution |
| `426a` quantile oracle | `7/11` | coverage + distribution | conditionality, coint, regime, MR |
| `427a` affine oracle | `5/11` | partial regime layer2 (`4/8`) | coverage, distribution, conditionality, coint, MR |
| `429a` path-shift oracle | `6/11` | conditionality, path geometry | coverage, regime, distribution, coint, MR |

## Mechanism Read

The failure is now clean:

1. Horizon/cell marginal corrections can force the unconditional level law to match, but
   they are not compatible with the conditional transition structure.
2. Path-level center correction preserves more of the residual geometry, but it cannot
   satisfy the horizon/regime coverage geometry.
3. The raw `392a` structural passes have thin margins: conditionality and worst-cell
   cointegration are close to their gates, so post-hoc output surgery easily breaks
   them.

This closes oracle calibration as a primary route to a publishable learned model. It is
useful as a diagnostic, but not as a model story.

## Decision

Do not keep adding calibration knobs.

The next paradigm must learn the joint future law directly rather than repairing samples
after generation. The cleanest remaining direction is a single-stage conditional
likelihood model over the 30-day future path:

- history encoder;
- future path density model;
- exact or tractable likelihood objective;
- no posterior/prior scaffold;
- no low-rank decoder;
- no bounded idio path;
- no post-hoc evaluator-specific correction.

This differs from the failed direct path-FM branch because the objective should be
likelihood/calibration-native rather than velocity matching plus sampler temperature.
It also differs from the AR branch because level occupancy is modeled jointly, not left
as an emergent property of recursive one-day transitions.

## Next Experiment

Select one minimal conditional normalizing-flow baseline over flattened future paths.

Success criterion:

- beat the `392a` `8/11` frontier, or
- demonstrate a clean mechanism that combines `392a` structural behavior with native
  level occupancy.

Failure criterion:

- if exact-likelihood joint future density repeats the direct-path failures, close this
  line and revisit the suite/product target rather than adding more knobs.

