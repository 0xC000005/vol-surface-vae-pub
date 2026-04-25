# 505a MixLinear Path Result

## Context

505a tested the architecture-side implication from the recent time-series
literature: a very small linear-mixer future-path backbone can sometimes beat
larger sequence models. The implementation added `mixer_type="mixlinear"` to the
existing empirical normal-score full-path flow so the official `339a` native
path evaluator could be reused.

The model has `67,849` parameters, versus roughly `850k` for the current 392a
frontier. It uses separable linear mixing across future time, cells, and channel
features, with history projected into the future path.

## Result

- Core change: `diffusion/block_ar/empirical_normal_score_path_flow_matching.py`
- Trainer: `experiments/backfill/block_ar/train_339a_empirical_normal_score_path_flow.py`
- Model: `models/backfill/505a_mixlinear_path_fm_s42/best_model.pt`
- Full suite: `results/block_ar/505a_mixlinear_path_fm_s42/full11.json`
- Score: `2/11`
- Passed: `surface`, `block_ar`
- Failed: `coverage`, `conditionality`, `time_series`, `cointegration`,
  `regime_coverage`, `distributional_fidelity`, `cross_cell_correlation`,
  `mean_reversion`, `pathwise_jump_realism`

Key metrics:

- Parameters: `67,849`
- Coverage90: `0.7640`
- Conditionality MAE reduction: `2.29%`
- Daily-change KS: `10/25`
- Level KS: `2/25`
- Median-bias fraction: `16/25`
- Bias magnitude: `24/25`
- Regime layer2: `0/8`
- Cointegration worst-cell ratio: `0.171`
- Cross-cell corr ratio: `0.376`
- Rank ratio: `3.337`
- Mean-reversion ratio: `0.917`
- Path max-jump KS: `0.573`

## Mechanism Read

The small linear mixer is trainable and fast, but it does not preserve the
financial surface law. It over-fragments cross-cell structure, producing too
high an effective rank and too low a mean correlation. It also fails daily-change
KS, level KS, move-size profile, and path max-jump realism.

This falsifies the naive reading of the MixLinear/Minkowski-linear clue for this
problem: parameter efficiency alone is not the missing mechanism. The model
needs a stronger shared latent geometry or AR transition prior to keep surfaces
coherent.

## Decision

Close minimal MixLinear direct-path flow as below-frontier. Do not scale this
exact architecture by layers or width as the next default step; that would
abandon the parameter-efficiency hypothesis without explaining the failure.

The next clean route is a post-experiment architecture analysis: decide whether
to add the one allowed core bias, a narrow learned bottleneck/shared-factor
geometry, or return to the 392a AR transition core and stop direct-path new-core
experiments.
