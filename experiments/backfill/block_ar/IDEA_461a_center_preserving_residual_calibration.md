# Autoresearch 461a: Center-Preserving Residual Calibration

## Context

The explicit-density reset is now falsified enough to stop deepening it blindly:

- `343a` scalar chain-rule NLL density scored `3/11`.
- `460a` day-vector full-covariance Gaussian NLL density scored `2/11`.
- Both are deployable likelihood models, but both learn weak conditional
  structure relative to the `392a` rollout-energy flow model.

The current deployable frontier remains `392a` at `8/11`. It already preserves
daily-change KS, cross-cell structure, mean reversion, pathwise realism, and
conditionality. Its remaining failures are coverage, regime-cell coverage, and
level distributional fidelity.

## Hypothesis

Return to the strongest learned conditional sampler (`392a`) and separate the
roles cleanly:

1. base learned conditional law: `392a`;
2. deployable calibration layer: fit only on pre-validation windows;
3. calibration acts on residuals around the model's conditional center, not on
   unconditional levels.

For a history `H`, base samples `X`, and base conditional center `m(H)`, transform

```text
X_cal = m(H) + a[horizon, cell, optional regime] * (X - m(H))
```

or a monotone residual quantile variant around the same center. The important
constraint is center preservation: do not move the conditional median/mean unless
future evidence shows a center-bias problem. This is different from the 456-458
level-quantile maps, which were marginal level corrections and weakened
conditionality.

## Why This Is Principled

The suite says the learned law is structurally good but interval/regime calibrated
poorly. A center-preserving residual calibration is a standard risk-system
separation:

- the neural generator learns the conditional scenario geometry;
- the calibration layer controls interval reliability using held-out historical
  residuals;
- the center is preserved, so conditional signal, mean reversion, and correlation
  structure should be disturbed less than with level maps.

This is not a new architecture knob. It is a deployable policy calibration step
that should be reported separately from base-model metrics.

## First Falsifier

Fit a residual scale map on the pre-validation calibration panel:

- generate base `392a` samples for each calibration history;
- compute the sample center per history/horizon/cell;
- choose residual scale factors that move empirical 90% coverage toward the gate
  while respecting the upper overcoverage cap;
- apply the same center-preserving transform to validation samples;
- evaluate the unchanged 11-suite.

If conditionality remains above gate and coverage/regime improve, continue with a
residual quantile variant. If conditionality or structure collapses, calibration
around `392a` is not the route to a deployable 11/11 system.
