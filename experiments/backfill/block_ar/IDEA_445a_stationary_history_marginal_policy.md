# 445a: Stationary History-Marginal Policy

## Motivation

After `444a`, learned fine-tuning is not safely improving the `392a` frontier. However,
one of the persistent failures, level KS, is tied to the suite's stationarity framing:
aggregated generated future levels should match the historical marginal level law.

If that is the requirement, a deployable risk policy may use the observed pre-validation
history-level marginal as a stationary prior. This does not use validation futures and
does not fit per-validation-window corrections.

## Difference From Failed Calibration Branches

This is not a residual-error bank:

- no sampled calibration future errors;
- no nearest-neighbor residual transfer;
- no per-window target fitting.

It is also not the `403a` future-quantile map:

- `403a` maps generated samples to pre-validation future levels;
- `445a` maps generated samples toward pre-validation observed history levels, because
  the level-KS test itself is a stationarity sanity check over rolling windows.

## Proposed Falsifier

Run one conservative stationarity map around frozen `392a`:

1. Fit generated quantiles from `392a` calibration samples.
2. Fit target quantiles from pre-validation observed history levels, pooled across
   history time and windows.
3. Apply a partial monotone quantile map to validation samples with a fixed modest
   alpha.
4. Evaluate the unchanged full 11-suite and report the result as a calibrated risk
   system, not as the base learned law.

## Decision Rule

If this does not beat `392a` and improve level KS without causing new structure failures,
then the stationarity-calibration route should also be considered capped.
