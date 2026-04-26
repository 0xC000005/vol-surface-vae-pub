# 555a Local-Factor Conditioned Core Result

## Context

554a showed that local non-IV factors predict validation failure geometry, but a post-hoc factor score does not transfer out of sample. 555a therefore tested the clean core version: start from the current 510a/340c frontier checkpoint, add observed local factor history from `vol_surface_with_ret.npz` (`ret`, `price`, `slopes`, `skews`, `levels`), and fine-tune the full empirical-normal-score flow-matching core on the 441 pre-validation windows.

## Implementation

- Added local factor-history utilities in `_local_factor_conditioning_555_utils.py`.
- Added `train_555a_local_factor_core_finetune.py`.
- Added `evaluate_555a_local_factor_core_finetune.py`.
- Source checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`.
- Train scope: all core parameters, not only the factor side-channel.
- Training: 441 pre-validation windows, 4 epochs, `lr=1e-4`, local factor dim `5`.

The learned factor context scale remained tiny (`0.00046` by epoch 4), so this run should be read as a weakly factor-conditioned core fine-tune rather than evidence that the model is strongly using the local factors.

## Full-Suite Result

Score: `8/11`.

Failed suites:

- `coverage`
- `regime_coverage`
- `distributional_fidelity`

Key metrics:

- cov90 overall: `0.873`
- lower-only coverage pass: `true`
- conditionality MAE reduction: `6.20%`
- cointegration ratio: `0.705`
- cointegration worst-cell ratio: `0.263`
- regime layer2: `0/8`
- regime lower-only worst cell: `0.385`
- daily-change KS: `25/25`
- level KS: `9/25`
- median-bias cells: `21/25`
- mean-reversion: pass
- pathwise max-jump KS: `0.346`

## Risk-Readiness Read

Under the risk-manager framing from 552a, 555a ties 510a at stress score `3/4`; both fail regime lower-only inclusion.

Compared with 510a:

- 555a improves conditionality (`6.20%` vs `5.12%` MAE reduction).
- 555a preserves scenario authenticity, cointegration, mean reversion, and pathwise realism.
- 555a does not improve deployability because regime worst-cell under-inclusion worsens (`0.385` vs `0.538`).
- 555a also has slightly weaker level KS (`9/25` vs `10/25`).

## Mechanism Read

Local factor conditioning is mechanically safe and can improve conditionality, but this implementation does not solve the deployability blocker. The remaining failure is not "missing a factor feature" in a simple conditioning sense; it is the future IV-level/regime occupancy geometry. The factor context is too weakly used, and full-core FM fine-tuning from 510a does not automatically allocate enough probability mass to the sparse regime/cell stress states.

## Decision

Keep 555a as evidence that local factors can be integrated cleanly, but do not promote it over 510a as the risk-manager prototype. The next HEAD step should analyze the remaining regime under-inclusion directly and decide whether the next clean move is a learned stress-state objective, a scenario-set selection objective, or a broader paradigm shift. Do not add another factor side-channel or post-hoc factor calibration knob.
