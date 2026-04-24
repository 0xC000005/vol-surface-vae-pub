# 397a Target-Aligned Calibration Objective Ideation

## Context

The 391-396 path-energy family is closed as a primary route. The decisive falsifier is
395a: it had the best internal validation total and best validation energy, but the worst
official score among the energy runs. This means the next move should not be another
energy-weight, sample-count, or architecture tweak.

## First-Principles Requirement

A conditional scenario generator must satisfy this identity:

`Y | history ~ F_theta(. | history)`, so the predictive CDF value
`U = F_theta(Y | history)` should be uniform over held-out history/outcome pairs.

This is the direct calibration statement behind coverage, marginal occupancy, and
level-distribution fidelity. It is not IV-specific and it does not require a low-rank
decoder, regime head, post-hoc correction, or deterministic center/residual split.

## Evidence From This Repo

- Path-energy is valid but misaligned with the full target: 395a optimized the energy
  holdout objective and regressed official conditionality and cointegration.
- Earlier calibration-loss work showed that direct coverage objectives can move the
  intended sub-metrics. In particular, 251g's soft window-coverage loss moved the
  window-floor metric from failing to passing in that older family.
- Earlier post-hoc calibration heads are not the right template here: they added an
  output correction module and created calm/turb tradeoffs. The clean variant should
  fine-tune the generative core itself with a calibration objective, not attach a
  separate correction layer.

## Proposed Family

Keep the current single generative core unchanged and add a sampled calibration loss:

1. Generate multiple free-running paths from the current conditional law.
2. Convert sampled normal-score paths back to IV space using the model's empirical
   quantile map.
3. Estimate a soft PIT value per observed future point:
   `u = mean_k sigmoid((y - sample_k) / tau)`.
4. Penalize non-uniform PIT moments over the training batch:
   mean close to `0.5`, variance close to `1/12`.
5. Keep the ordinary flow-matching loss as an anchor.

This is target-aligned with the tests but remains architecturally clean. It asks the
same model to learn the conditional law more accurately rather than adding a risk-policy
calibration layer.

## Decisive Next Experiment

Implement one recent-window soft-PIT fine-tune starting from the active best 392a:

- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- no architecture changes
- loss: `FM anchor + lambda_pit * soft_PIT_moment_loss`
- first probe: small `lambda_pit`, `n_samples=8`, recent 441-window adaptation block

Kill condition: if official validation does not improve beyond 8/11, or if it regresses
conditionality/cointegration like the energy family, close sampled calibration fine-tuning
and move to a paradigm-level objective review.
