# 633a Native Joint Rollout Proper-Scoring Ideation

## Context

631a is the active native joint base: one state-conditioned transition model over encoded daily increments, with generic scale-prefix features and deterministic integration back to levels. It is mechanically compatible with both IV-only and 25+13 joint panels, but the IV suite remains at 4/11 because multi-step scenario calibration is weak: coverage, regime/turbulent width, distributional fidelity, mean reversion, surface stability, and jump realism remain fragile.

632a falsified the local conditional source-scale fix. The source-scale head collapsed to the lower bound under the one-step flow-matching objective and damaged free-running regime and dependence behavior. This means the bottleneck is not another local noise-scale knob; it is the mismatch between the one-step training objective and the 30-day scenario-set evaluation.

## Hypothesis

The next principled move is to keep the 631a architecture and train it with a rollout-level proper scoring objective. Energy score, patch-energy score, and sliced projection Wasserstein are proper or distributional scoring rules over generated scenarios. They target the object the risk manager consumes: a conditional set of future paths, not just a one-step conditional increment.

## Evidence From Prior Branches

- The historical best learned IV models, 392a and 510a, reached the 8/11 frontier using rollout/patch-energy fine-tuning rather than purely one-step likelihood or one-step flow matching.
- 596a already implements a full-path energy plus sliced projection Wasserstein objective and includes differentiable tests for these losses.
- The native joint branch from 627a-632a shows that encoded increments plus state conditioning are the right coordinate system for 25+13 generation, but one-step objectives still miss free-running path calibration.
- `sample_batched(...)` for the 631a model is intentionally `no_grad`, so this must be implemented as a small differentiable rollout trainer/helper rather than by wrapping the evaluator or adding post-hoc deck calibration.

## Clean-Pathology Decision

Do not add another architecture branch, bounded path, IV/factor-specific correction, or source-scale knob. Keep the generative core and state-space coordinate fixed:

- Model object: one generic state-conditioned encoded-increment transition law.
- Generated object: encoded daily increments for all panel channels.
- Conditioning object: recent encoded levels, encoded increments, and generic scale-prefix statistics.
- Objective addition: small rollout-level proper score on generated future paths, anchored by the original one-step flow-matching loss.

## Next Experiment

Run 634a as a minimal rollout-proper-scoring fine-tune from 631a:

- Add a differentiable rollout helper for `GenericStateConditionedIncrementFlowMatching`.
- Train with `FM anchor + small full-path energy / sliced-Wasserstein score` over the generated encoded-increment or reconstructed encoded-level path.
- Use all 38 channels when training `joint38`; do not split IV and anchor factors into separate models.
- Evaluate with the existing IV full 11-suite and the joint-panel audit.

The acceptance criterion is not simply higher train loss quality. It must improve risk-manager scenario behavior without losing the clean native joint mechanism: IV full-suite score should improve over 631a 4/11, and joint-panel dependence/tail metrics should remain close to 629a/631a rather than collapsing like 632a.
