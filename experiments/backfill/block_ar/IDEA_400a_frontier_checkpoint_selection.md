# 400a Frontier Checkpoint-Selection Review

## Context

The active frontier is still 392a at 8/11. The recent objective probes are closed:

- path-energy fine-tuning is capped because better internal energy geometry regresses
  official conditionality/cointegration before level KS reaches the gate;
- low-order soft-PIT fine-tuning is closed because it improves internal rank moments
  while damaging level law, kurtosis, and worst-cell cointegration.

The common lesson is not "add another loss." It is that internal validation losses are
not aligned enough with the final full-11 target.

## Review

The remaining 392a failures are:

- coverage: a small number of per-cell over/under cases, mostly edge/horizon imbalance;
- regime coverage: layer2 remains 0/8, and this is partly a conservative risk-policy
  calibration requirement over small calm/turb subsamples;
- distributional fidelity: daily-change KS is solved, median/bias magnitude mostly pass,
  but level KS remains below the 15/25 gate.

These failures are not solved by the last two objective families, but they also do not
prove that the 392a training trajectory's validation-selected checkpoint is the best
official checkpoint. 396a already showed that internal validation objective ordering can
disagree with official 11-suite ordering.

## Selected Next Step

Before opening another large architecture or likelihood paradigm, evaluate the existing
`final_model.pt` from the clean 392a weak path-energy trajectory:

`models/backfill/392a_recent_rollout_energy_w005_s42/final_model.pt`

This is not a new model knob. It is a checkpoint-selection audit of an already completed,
clean trajectory. The falsifier is simple:

- if the final checkpoint improves beyond 8/11 or materially improves failed residuals
  without losing core passes, continue checkpoint/proxy-selection work;
- if it is worse or equivalent, close this route and move to a true paradigm-level review.

## Why This Is Principled

The suite has repeatedly shown that ordinary holdout losses are not reliable selectors.
Checking an already trained endpoint is a minimal, non-architectural test of that finding.
It preserves the current methodology discipline: no new architecture, no new loss, no
post-hoc calibration, and no evaluator-specific optimization.
