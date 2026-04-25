# 504a Joint Sliced-Wasserstein Result

## Context

504a tested a literature-informed objective swap after the 503a new-core audit. The
motivation was DistDF/MMPD-style distributional training: keep the deployable 392a
AR flow core unchanged, but replace the already-falsified joint RBF-MMD alignment
with a joint sliced-Wasserstein alignment over `(history, future)` IV paths.

This is not a calibration wrapper and not a new architecture. It is a direct
learned-law objective test.

## Result

- Trainer: `experiments/backfill/block_ar/train_504a_recent_joint_sliced_wasserstein_finetune.py`
- Source: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- Model: `models/backfill/504a_recent_joint_sw_w005_s42/best_model.pt`
- Full suite: `results/block_ar/504a_recent_joint_sw_w005_s42/full11.json`
- Score: `7/11`
- Failed: `coverage`, `cointegration`, `regime_coverage`, `distributional_fidelity`

Key metrics:

- Coverage90: `0.8566`
- Conditionality: pass, MAE reduction `5.2%`
- Daily-change KS: `25/25`
- Level KS: `11/25`
- Median-bias fraction: `19/25`
- Bias magnitude: `24/25`
- Regime layer2: `0/8`
- Cointegration worst-cell ratio: `0.211`
- Cross-cell corr ratio: `0.938`
- Mean-reversion ratio: `1.015`
- Path max-jump KS: `0.379`

## Mechanism Read

The sliced-Wasserstein objective is active and does not collapse the rollout law,
but it does not create a new deployable frontier. Relative to 392a, it nudges
level KS from `10/25` to `11/25` while losing worst-cell cointegration and keeping
regime layer2 at `0/8`.

This is the same core tradeoff seen in stronger energy, MMD, CRPS, PIT, and
calibration variants: distribution-alignment pressure can move unconditional
level occupancy slightly, but the movement is not enough to pass distributional
fidelity and it erodes at least one structural gate.

## Decision

Close weak joint sliced-Wasserstein fine-tuning as below-frontier. Do not run a
scalar SW-weight sweep as the next default step; that would be knob search unless
there is a sharper mechanism.

The next principled branch should use the new literature in the architecture
direction rather than another local loss around 392a: a minimal
MixLinear/Minkowski-linear style efficient future-path backbone, evaluated as a
new learned core against the 392a frontier.
