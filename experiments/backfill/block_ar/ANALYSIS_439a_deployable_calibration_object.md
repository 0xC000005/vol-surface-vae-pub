# 439a: Deployable Calibration Object Analysis

## Question

After `438a`, should the deployable path continue, and if so what object should be
calibrated?

## Evidence

- `392a` remains the learned-core frontier at `8/11`.
- `403a` empirical quantile mapping was deployable but fell to `6/11`; it changed
  marginal levels too aggressively and broke conditionality/time-series gates.
- `405a/407a` interval scaling was deployable and preserved more structure, but stayed
  at `6-7/11`; it could not fix level distribution or regime layer2 coverage.
- `438a` global residual-error bootstrap was deployable but scored `7/11`; it preserved
  pathwise jump realism and tail scale, but worsened level KS (`4/25`) and left coverage
  and regime layer2 unresolved.
- `435a` reached `11/11` only by using validation futures as the center, so it proves
  suite feasibility but not deployability.

## Mechanism Read

The failing object is not just interval width. It is the conditional future center/path
law. Width-only calibration cannot move enough mass to repair level KS. Global residual
calibration moves mass, but because it is not sufficiently conditional on the current
history, it imports forecast-error paths from incompatible states and damages level
marginals and small-move realism.

This explains the current pattern:

- `392a`: good structure, insufficient calibrated coverage/level law.
- width calibration: preserves structure, does not solve center/path-law error.
- global residual bank: adds realistic movement, but too unconditional.
- oracle center: solves all tests, but is not deployable.

## Next Clean Falsifier

The next deployable calibration object should be a history-local residual-error law:

1. Fit `392a` forecast errors on pre-validation calibration windows.
2. Represent each calibration history with generic panel-history features, not IV-specific
   labels: last surface, recent average surface, recent change, and realized history
   variance.
3. For each validation history, retrieve a fixed number of nearest calibration histories
   in that feature space.
4. Sample residual-error paths only from that local calibration neighborhood.
5. Add a small `392a` residual-shape term to preserve the learned core's sample geometry.

This is not a new generative core. It is a split-calibration policy around the frozen
learned core, using only pre-validation outcomes and validation histories. It remains
deployable if the neighbor bank is frozen before validation evaluation.

## Falsification Rule

Run exactly one local-residual experiment first.

Success means it beats the `392a` deployable frontier and moves toward the remaining
coverage/regime/distribution failures without creating new time-series or structure
failures.

Failure means the deployable residual-calibration route is probably not sufficient, and
the next principled move should return to improving the learned conditional center/path
model rather than adding calibration knobs.
