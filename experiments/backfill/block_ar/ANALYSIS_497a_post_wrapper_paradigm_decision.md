# 497a Post-Wrapper Paradigm Decision

## Context

The 392a frontier remains the strongest deployable learned conditional law at `8/11`.
It passes the structural suites and conditionality, but fails coverage, regime
coverage, and distributional fidelity. The recent repair families did not improve the
frontier:

- Calibration tables and residual maps (`456-466`) moved coverage symptoms but broke
  conditionality, time-series shape, cointegration, or level law.
- Critic, density-ratio, source-transport, and proper-score branches (`467-494`) were
  capped below `392a`.
- Learned stress ensembling (`495-496`) preserved some structural diagnostics but
  diluted conditionality and still missed level/regime allocation.

## Suite Read

The suite is not internally contradictory after the recent audit:

- Coverage and regime coverage require per-cell 90% intervals to stay inside
  `[70%, 95%]`, so both undercoverage and broad overcoverage are penalized.
- The turbulent/calm width target is now informational, not a hard learned-law gate.
- Pathwise max-jump KS is relaxed to `0.50`, so pathwise realism is no longer the
  binding failure for `392a`.
- Distributional fidelity still requires level KS `>=15/25`; `392a` is `10/25`, while
  recent wrappers top out near `13/25`.

The remaining hard failures are all manifestations of conditional level/allocation
calibration, not missing local path geometry.

## Decision

Close the 392a wrapper/local-repair family as a pure learned-law route. A deployable
`11/11` system now requires a separated risk-policy layer:

```text
base learned law:     392a conditional path generator
risk-policy layer:    deployable calibration of sample allocation/intervals
reporting rule:       report base-model metrics and final calibrated-system metrics
```

This is not a Bitter-Lesson claim about the learned core. It is a risk-management
system claim: the learned generator supplies the conditional path geometry, while a
pre-validation policy calibrator enforces conservative per-cell/per-regime interval
allocation without using validation futures.

## Next Experiment

Implement one constrained policy-calibration falsifier rather than another learned
architecture:

- preserve the 392a conditional median as much as possible to protect conditionality;
- operate only on the residual distribution around that median;
- fit calibration statistics on pre-validation windows only;
- use no validation future, no oracle center, and no evaluator-specific per-window
  ground truth;
- evaluate as a deployable system and report it separately from `392a`.

If this cannot exceed `8/11`, the honest conclusion is that the current deployable
system is `392a` plus failed policy-calibration attempts, not an undiscovered neural
architecture issue.
