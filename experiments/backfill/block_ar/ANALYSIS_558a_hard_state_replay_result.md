# 558a Hard-State Replay Patch-Energy Result

## Context

557a selected the smallest clean move after 556a: keep the 510a empirical-normal-score core, compute calibration-only hard-state replay weights from frozen-source undercoverage, and fine-tune with a patch-energy objective reweighted toward hard windows.

The aim was to target sparse regime/cell under-inclusion without global widening, factor side-channels, or evaluator-time per-cell tables.

## Implementation

- Added `train_558a_hard_state_replay_patch_energy.py`.
- Added `test_code/test_558a_hard_state_replay.py`.
- Source checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`.
- Calibration block: 441 pre-validation windows.
- Frozen-source hard-score sampling: 32 samples per window.
- Replay weights: rank-smooth mean-one weights, min `0.600`, max `1.799`.
- Training: 4 epochs, best checkpoint at epoch 1.

Hard-state summary on calibration:

- mean 90% coverage: `0.845`
- mean upper miss rate: `0.090`
- mean lower miss rate: `0.064`
- hard score p90: `0.465`
- hard score max: `1.231`

## Full-Suite Result

Score: `7/11`.

Failed suites:

- `coverage`
- `cointegration`
- `regime_coverage`
- `distributional_fidelity`

Key metrics:

- cov90 overall: `0.874`
- h30 cov90: `0.889`
- conditionality MAE reduction: `5.96%`
- turb/calm width ratio: `1.112`
- cointegration ratio: `0.560`
- cointegration worst-cell ratio: `0.139`
- regime layer2: `0/8`
- risk lower-only regime worst cell: `0.513`
- daily-change KS: `25/25`
- level KS: `11/25`
- mean-reversion: pass
- pathwise max-jump KS: `0.320`

Risk-readiness score: `1/4`.

## Mechanism Read

Hard-state replay moved the model in the intended direction on some marginal geometry:

- level KS improved to `11/25`;
- pathwise realism stayed strong;
- conditionality stayed above the hard gate.

But it did not repair risk deployability:

- localized regime under-inclusion remained;
- worst-cell coverage fell just below the lower-only coverage gate;
- cointegration worst-cell ratio collapsed to `0.139`, breaking scenario authenticity.

This is the same recurring tradeoff as prior proper-score variants: extra pressure on hard distributional states can move marginal level occupancy, but it erodes a structural dependence gate before regime coverage becomes acceptable.

## Decision

Close this exact hard-state replay patch-energy implementation as below-frontier. The next move should not be another scalar replay-weight sweep. If continuing this family, the next clean variant must change the sampling law itself, such as a small routed source-prior expert, rather than reweighting the same patch-energy loss.
