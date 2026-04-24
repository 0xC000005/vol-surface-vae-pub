# 402a Calibrated Risk-System Paradigm

## Context

The learned base-law frontier is still 392a at 8/11. Several clean base-law repair
attempts are now closed:

- stronger path-energy weights improve level occupancy but lose conditionality or
  cointegration;
- better energy Monte Carlo estimation improves the internal objective but worsens the
  official suite;
- low-order soft-PIT fine-tuning improves internal rank moments but damages level law,
  kurtosis, and worst-cell cointegration;
- endpoint checkpoint selection improves coverage edge counts but loses cointegration.

## Decision

Open a separate calibrated-risk-system branch.

This is not a claim that the base model learned the full conditional law. It is a
production/risk-management system layer:

1. Base model: report 392a metrics as the learned conditional generator.
2. Calibration layer: report separately as policy calibration fit on pre-validation
   history/outcome pairs.
3. Final system: evaluate base + calibration for risk-manager usability.

## Calibration Object

Use monotone empirical quantile calibration in IV space:

- sample the frozen 392a base model on a pre-validation adaptation block;
- for each horizon/cell, estimate the generated marginal CDF and the realized marginal CDF;
- map each generated value through generated CDF then target inverse CDF;
- optionally condition the map on history vol-of-vol bins to support regime coverage.

This is a monotone output transform, not a new decoder, not a learned posterior/prior, and
not a low-rank or bounded residual architecture. It is explicitly a statistical policy
layer on top of the base generator.

## Why This Is Acceptable

For a risk manager, calibrated scenario intervals are a valid product requirement. The
paper must not present this as the base model's learned conditional law. The defensible
presentation is:

- learned base model: 8/11, clean architecture, conditional-law evidence;
- calibrated risk system: post-training statistical calibration, evaluated separately;
- ablation: show what calibration changes and what it cannot claim.

## Next Experiment

Implement one standalone evaluator for 392a plus pre-validation marginal/regime quantile
calibration. Falsifier:

- if it cannot improve beyond 8/11, this route is not enough;
- if it improves by fixing level KS/coverage/regime without destroying conditionality,
  continue as the calibrated system track;
- if it reaches 11/11, mark the goal as final-system achieved but keep base and calibrated
  metrics separated in all documentation.
