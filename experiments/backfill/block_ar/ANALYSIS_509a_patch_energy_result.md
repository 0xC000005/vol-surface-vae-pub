# 509a Patch Energy Result

## Context

508a selected one final MMPD-inspired local objective falsifier: keep the 392a
AR transition core unchanged, retain the FM anchor, and add an energy score over
overlapping 5-day future patches.

## Result

- Trainer: `experiments/backfill/block_ar/train_509a_recent_patch_energy_finetune.py`
- Source: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- Model: `models/backfill/509a_recent_patch_energy_l5_w005_s42/best_model.pt`
- Full suite: `results/block_ar/509a_recent_patch_energy_l5_w005_s42/full11.json`
- Score: `6/11`
- Passed: `surface`, `time_series`, `block_ar`, `cross_cell_correlation`,
  `mean_reversion`, `pathwise_jump_realism`
- Failed: `coverage`, `conditionality`, `cointegration`, `regime_coverage`,
  `distributional_fidelity`

Key metrics:

- Coverage90: `0.8736`
- Conditionality MAE reduction: `3.88%`
- Daily-change KS: `25/25`
- Level KS: `10/25`
- Median-bias fraction: `20/25`
- Bias magnitude: `25/25`
- Regime layer2: `0/8`
- Cointegration worst-cell ratio: `0.211`
- Cross-cell corr ratio: `0.970`
- Rank ratio: `1.463`
- Mean-reversion ratio: `0.965`
- Path max-jump KS: `0.388`

## Mechanism Read

Patch energy is cleaner than the failed direct-path cores. It preserves local
time-series realism, cross-cell structure, mean reversion, and pathwise jumps.
It also passes median-bias fraction and bias magnitude.

But it is still below the 392a frontier. It does not improve level KS beyond the
392a `10/25`, regime layer2 remains `0/8`, and the extra patch-distribution
pressure weakens conditionality and worst-cell cointegration. This is the same
frontier tradeoff in another form: local distributional pressure can improve
some calibration diagnostics, but not the coupled level/regime allocation
without losing structural gates.

## Decision

Close MMPD-inspired local patch objective as below-frontier. Do not sweep patch
length or patch-energy weight. The literature-derived local tests have now been
covered: joint Wasserstein, efficient linear mixer, shared source geometry, and
patch energy all remain below 392a.

The next step should be a synthesis/closure decision unless a genuinely new
data source, evaluation framing, or larger new-core program is introduced.
