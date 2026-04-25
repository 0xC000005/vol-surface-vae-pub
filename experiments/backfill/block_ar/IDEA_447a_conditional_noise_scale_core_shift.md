# 447a: Conditional Noise-Scale Core Shift

## Why Shift

The wrapper routes around `392a` are capped:

- residual-error calibration (`438a`, `440a`) damages level law or conditionality;
- stationary marginal mapping (`446a`) improves coverage but weakens conditionality and
  cointegration;
- learned checkpoint ensembling (`442a`) dilutes conditional accuracy;
- direct quantile/coverage fine-tuning (`444a`) damages level bias, tails, and
  cointegration.

The repeated failure is width allocation inside the learned conditional law. `392a`
already has a good conditional center/path structure, but it cannot allocate uncertainty
across history regimes, cells, and horizons well enough to pass coverage/regime layer2.

## New Core Hypothesis

Make conditional entropy a learned part of the generative core by enabling the model's
existing conditional base-noise scale head.

This is not a post-hoc calibration layer. In flow matching, the generated transition is
defined by a source distribution and a learned transport. Allowing the source noise scale
to depend on the causal memory state is a principled way to model history-dependent
conditional uncertainty.

## Clean Falsifier

Run one conservative fine-tune from `392a`:

- enable `conditional_noise_scale`;
- initialize the new scale head at identity;
- train only the new `noise_log_scale` head first, leaving the learned transport fixed;
- use the same recent pre-validation window as `392a`;
- optimize FM anchor plus free-running path-energy score;
- evaluate the unchanged 11-suite.

This tests whether the missing component is conditional uncertainty allocation, without
changing the learned center/path transport.

## Decision Rule

Success means beating the `392a` frontier while preserving conditionality, time-series,
cointegration, and structure.

Failure means conditional scale alone is insufficient, and the next learned-core change
must modify the transport/backbone itself rather than only the source noise.
