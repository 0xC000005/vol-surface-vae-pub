# Autoresearch 454a: conditional joint-MMD rollout scoring

## Context

The recent objective falsifiers separate the failure modes:

- 392a score-space rollout energy preserves conditionality and path geometry but
  still fails coverage, regime-cell coverage, and level KS.
- 453a IV-space marginal CRPS improves average calibration but weakens
  conditionality and cointegration, and level KS remains poor.
- 452a larger teacher-forced FM capacity improves validation FM loss but worsens
  rollout scenario law.

The common problem is objective alignment. Marginal distribution pressure can
calibrate widths without learning the conditional law. Teacher-forced FM can
improve transition likelihood without producing the right free-running path law.

## Hypothesis

Train the generator with a rollout-level score on the joint distribution
`(history, future_path)` instead of only the marginal future distribution.

A kernel MMD on pairs `(H, Y)` is a clean first-principles target:

- It is generic across financial factor panels.
- It does not name regimes, tails, low-rank factors, or evaluator gates.
- Matching the joint law preserves conditional dependence because the history
  kernel gates which futures are compared.
- With an FM anchor, it can be used as a light free-running alignment objective
  rather than a replacement for the learned transition law.

## Proposed Falsifier

Fine-tune 392a with:

- the existing FM anchor,
- differentiable free-running rollout samples,
- a joint-MMD loss between real pairs `(history, realized_future)` and generated
  pairs `(history, sampled_future)`,
- fixed median-heuristic RBF bandwidths computed inside each batch and detached
  from gradients.

Expected read:

- If conditionality stays above the 5% gate and level/regime metrics improve,
  this branch is alive.
- If it behaves like marginal CRPS, the objective is still too weak/noisy for
  one-realization-per-history conditional training and the next paradigm must
  change the data framing or likelihood estimator.

## Deployability

The resulting sampler is unchanged at inference: it uses only history and random
source noise. The MMD objective uses only pre-validation training windows during
fine-tuning and introduces no posthoc calibration map.
