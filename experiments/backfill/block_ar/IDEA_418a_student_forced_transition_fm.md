# 418a: Student-Forced Transition FM For Native AR Path Occupancy

## Context

The active frontier is still `392a` at `8/11`. It is an empirical normal-score AR transition flow with strong local mechanics and the right structural suites:

- passes conditionality, time-series, block-AR, cointegration, cross-cell correlation, mean reversion, and pathwise jump realism,
- fails coverage, regime coverage, and distributional fidelity,
- specifically remains short on long-horizon level occupancy (`10/25` level-KS cells) and regime/cell coverage (`0/8` layer2 combinations).

The direct path-flow repair branch is now closed. `413a` showed that direct recent-score path flow can repair level occupancy (`20/25`) but loses structural coupling. `417a` showed that importing the stronger old Transformer path mixer still does not combine occupancy with conditionality, cointegration, regime response, and mean reversion.

## Hypothesis

The clean remaining mismatch is not the factorization itself; it is the distribution on which the transition model is trained.

`392a` is evaluated as a self-generated 30-day AR scenario generator, but its base flow objective is teacher-forced: each transition is trained under ground-truth prefixes. Weak rollout-energy fine-tuning helps, but it only scores generated paths globally. It does not directly teach the one-step transition law to recover correct next-step conditional distributions when its prefix already contains model-generated states.

Train the same AR transition flow on model-generated prefixes:

1. sample a short free-running prefix from the current model,
2. condition the transition FM on that generated prefix,
3. match the next realized future score from that off-policy state,
4. keep the original teacher-forced FM anchor so local one-day mechanics are preserved.

This is a general exposure-bias correction. It does not introduce a low-rank decoder, bounded idio path, post-hoc calibration table, evaluator-specific KS loss, or regime-specific policy.

## Why This Is Different From Prior Proper-Score Fine-Tunes

`391/392` and `410` used proper scores on sampled full paths. Those losses can move average path geometry, but they do not change the conditional training distribution of the transition operator itself.

Student-forced FM attacks the operator mismatch directly:

- teacher-forced FM learns `p(x_t | true prefix)`,
- rollout evaluation needs `p(x_t | generated prefix)`,
- student-forced FM adds training mass on `generated prefix` while preserving the same vanilla transition-flow core.

## Decisive Test

Run one conservative fine-tune from the 392a checkpoint:

- source: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- objective: teacher-forced FM anchor plus student-forced transition FM on generated prefixes
- no post-hoc interval scaling
- no architecture change
- evaluate with the unchanged official full 11-suite.

Success criterion: beat the 392a frontier or produce a clean mechanism improvement on level KS / regime layer2 without losing conditionality, cointegration, and mean reversion. If it fails in the same tradeoff pattern, close off-policy transition fine-tuning and move to a deeper base-likelihood paradigm rather than tuning weights.
