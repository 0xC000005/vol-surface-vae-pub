# 441a: Learned-Law Ensemble Paradigm Shift

## Why Shift

The deployable residual-calibration route is capped:

- global residual calibration (`438a`) was too unconditional and scored `7/11`;
- local residual calibration (`440a`) was more conditional but fell to `6/11`;
- interval and quantile calibration (`403a`, `405a`, `407a`) either damaged structure or
  could not solve level/regime failures.

The remaining bottleneck is the learned conditional center/path law, not a residual-bank
or interval-width correction.

## New Paradigm

Use an equal-weight ensemble of nearby learned conditional generators as the deployable
sample law.

This is not post-hoc calibration. It is model averaging over learned conditional laws:

- no validation futures;
- no per-window correction;
- no fitted residual bank;
- no validation-tuned mixture weights;
- no evaluator-specific target.

The ensemble uses additional learned model uncertainty instead of manual output
correction. This is more aligned with the Bitter Lesson than adding hand-designed
coverage patches, while remaining operationally simple.

## Evidence Behind the First Ensemble

Recent learned checkpoints expose complementary tradeoffs:

- `391a`: stronger level KS (`13/25`) but weaker coverage/cointegration.
- `392a`: best official frontier (`8/11`), strongest balanced structure.
- `393a`: similar level KS (`12/25`) but weaker conditionality.

The clean falsifier is therefore an equal `391a/392a/393a` sample ensemble. Equal
weights avoid validation fitting and make the result interpretable as learned model
averaging.

## Falsifier

Run `442a`:

- load `391a`, `392a`, and `393a`;
- draw `16` paths from each model for `48` total samples per history;
- evaluate the unchanged full 11-suite;
- report explicit provenance that no validation-future information or calibrated
  weights were used.

Success means the ensemble beats the `392a` deployable frontier and reduces the
coverage/regime/distribution failures without creating new failures.

Failure means simple learned-law averaging is insufficient, and the next learned-core
route must directly train a path-law objective rather than combine existing checkpoints.
