# 500a State-Conditional Center Policy Ideation

## Evidence

Recent policy experiments isolate the remaining bottleneck:

- `498a` median-locked asymmetric tail policy preserves conditionality but leaves level
  KS unchanged at `10/25`. Width-only policy is capped.
- `499a` global center quantile mapping targets level occupancy directly but fails badly:
  level KS drops to `1/25`, median bias worsens, mean reversion fails, and coverage
  undercovers later horizons.

This means the center/level law cannot be repaired by unconditional pre-validation
distribution matching. Any deployable center movement must be conditioned on the current
state and must be predictive, not just marginally distribution-matching.

## Principle

The cleanest remaining policy layer is a small state-conditional center residual model:

```text
base center      = 392a sample median
target residual  = realized future - base center
features         = low-dimensional history summaries
center policy    = ridge-predicted residual, selected by pre-validation holdout MAE
tail policy      = asymmetric residual scaling around shifted center
```

This is not a new neural architecture and should not be reported as base learned-law
progress. It is a deployable risk-policy calibration layer. The justification is
practical: risk managers need a usable scenario system, and the base learned law is
already the honest conditional generator frontier.

## Guardrails

- Use only pre-validation history/future pairs for fitting.
- Use validation futures only in the final suite.
- Keep the center model low-dimensional and linear/ridge to avoid another opaque
  architecture branch.
- Select any shrinkage only on a pre-validation holdout using predictive MAE, not suite
  gates.
- Preserve 392a residual path geometry as much as possible.
- Report base `392a` metrics separately from final calibrated-system metrics.

## Next Falsifier

Run `501a`: fit a state-conditional ridge center residual policy, shift the 392a sample
median by the predicted residual, then fit/apply asymmetric residual tails around the
shifted center.

Success condition:

- improve level KS above the `392a`/`498a` `10/25` level without losing conditionality
  and core structural suites.

Failure condition:

- if state-conditional center prediction also fails to improve level occupancy without
  structural damage, close center-policy calibration and return to the product decision:
  `392a` is the deployable learned frontier; `11/11` remains oracle-feasible but not
  deployably learned under current data/suite.
