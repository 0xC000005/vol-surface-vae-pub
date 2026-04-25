# 428a: Oracle Calibration Failure-Mode Analysis

## Context

The `392a` learned frontier remains `8/11`. Two validation-oracle calibration
diagnostics have now tested whether post-hoc output calibration can close the remaining
coverage/regime/distributional gap:

- `426a`: nonlinear marginal quantile map;
- `427a`: affine median-shift plus interval-scale map.

Both freeze the same `392a` generator.

## Results

| System | Score | Coverage | Distribution | Conditionality | Cointegration | Regime Layer2 | Mean Reversion |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `392a` raw | `8/11` | fail | fail | pass (`5.14%`) | pass (`0.278` worst) | `0/8` | pass (`83.3%`) |
| `426a` quantile oracle | `7/11` | pass | pass | fail (`4.70%`) | fail (`0.200` worst) | `1/8` | fail (`45.8%`) |
| `427a` affine oracle | `5/11` | fail | fail | fail (`4.83%`) | fail (`0.185` worst) | `4/8` | fail (`41.7%`) |

## Mechanism Read

The marginal corrections are attacking the right symptom but at the wrong object.

`392a`'s strongest evidence is structural: it preserves transition behavior, EWMA
coupling, cross-cell geometry, and mean reversion. The two oracle calibrations operate
per horizon/cell, so they improve level occupancy by changing the path law after the
model has generated it. That breaks the same structural signatures that made `392a`
valid.

This also makes a simple interpolation sweep unattractive. The structural margins are
thin:

- conditionality starts barely above the `5%` MAE-reduction gate and both full
  corrections fall below it;
- worst-cell cointegration starts at `0.278`, only slightly above the `0.25` gate, and
  both corrections push it below;
- mean-reversion active-cell pass rate collapses under both corrections.

A small interpolation would likely preserve structure but not repair level/regime
failures; a large interpolation repairs marginal suites but breaks structure. That is a
tradeoff curve, not a clean path to `11/11`.

## Decision

Close per-horizon/cell marginal oracle calibration as the primary route.

The next diagnostic should preserve path geometry more aggressively: move whole future
paths by a low-dimensional oracle center correction instead of remapping every
horizon/cell marginal. This tests whether the remaining gap is mainly a path-center
error while retaining generated residual dynamics.

The clean next experiment is:

- freeze `392a`;
- compute a validation-oracle path-level vertical center shift from the generated path
  median to the realized future;
- apply the same shift to all samples in that window, preserving residual path shape;
- evaluate the unchanged full 11-suite;
- treat any success as an oracle feasibility bound, not a deployable learned law.

