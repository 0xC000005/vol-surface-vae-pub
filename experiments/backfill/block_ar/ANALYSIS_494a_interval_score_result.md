# 494a Interval-Score Fine-Tune Result

## Hypothesis

Before closing the 392a repair family, test one proper scoring-rule objective
that directly targets calibrated central intervals while keeping the 392a
architecture and sampler unchanged.

494a fine-tunes the 392a checkpoint on recent pre-validation windows with:

- original teacher-forced FM loss as anchor,
- free-running rollout samples in empirical-score coordinates,
- central 90% interval score with `interval_weight=0.02`,
- no posthoc per-cell/regime tables or evaluator-time calibration.

## Artifacts

- Trainer: `experiments/backfill/block_ar/train_494a_recent_interval_score_finetune.py`
- Model: `models/backfill/494a_recent_interval_score_w002_s42/best_model.pt`
- Train summary: `models/backfill/494a_recent_interval_score_w002_s42/train_summary.json`
- Full suite: `results/block_ar/494a_recent_interval_score_w002_s42/full11.json`
- Markdown: `results/block_ar/494a_recent_interval_score_w002_s42/full11.md`

## Result

Score: `6/11`.

Passed:

- surface
- block_ar
- cointegration
- cross_cell_correlation
- mean_reversion
- pathwise_jump_realism

Failed:

- coverage
- conditionality
- time_series
- regime_coverage
- distributional_fidelity

Key metrics:

- coverage90: `0.9092`
- conditional MAE reduction: `4.93%` versus gate `>5%`
- turb/calm width ratio: `1.147` informational, just below `1.15`
- daily KS: `25/25`
- level KS: `4/25`
- median-bias pass cells: `16/25`
- regime layer2: `1/8`
- cointegration worst-cell ratio: `0.263`
- cross-cell corr ratio: `0.995`
- mean-reversion ratio: `0.958`, active pass `83.3%`
- pathwise max-jump KS: `0.275`, q99 pass `22/25`

## Mechanism Read

The interval score did what it is designed to do, but that is not enough:

- It increased interval width and eliminated most undercoverage, raising overall
  90% coverage to `90.9%`.
- The per-cell upper cap still fails because several cells exceed 95% coverage;
  regime layer2 improves only to `1/8`.
- Conditionality narrowly fails at `4.93%`, matching the recurring pattern where
  calibration pressure weakens the history-conditioned advantage.
- Distributional fidelity worsens: level KS falls to `4/25` and median-bias pass
  cells fall to `16/25`, despite daily-change KS staying perfect.

## Decision

Close this exact proper-score repair as below-frontier. 494a confirms the same
cap observed in energy, marginal-CRPS, critic, density-ratio, source-transport,
and marginal-map branches: objective pressure can move average calibration but
does not learn the missing conditional level/regime allocation without damaging
another required property.

The next step should be a paradigm decision, not another 392a-local objective
weight. Either define a genuinely new core that models future level allocation
jointly with path dynamics, or explicitly accept that the current test target
requires non-learned policy calibration beyond the base model.
