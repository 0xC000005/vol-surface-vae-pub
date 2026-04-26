# 561a TimePFN-Style Synthetic Prior Scaled Result

## Context

560a implemented the synthetic-prior pretraining scaffold and proved the plumbing works, but the tiny smoke checkpoint was not a candidate model. 561a scaled the same idea to the 340c/392a-sized empirical-score AR flow backbone.

## Run

Synthetic pretraining:

- Output: `models/backfill/561a_timepfn_synthetic_prior_scaled`
- Synthetic windows: `4096`
- Epochs: `8`
- Parameters: `850,561`
- Best validation loss: `0.2106`

Real-data adaptation:

- Output: `models/backfill/561a_timepfn_synthetic_prior_scaled_adapt`
- Source checkpoint: `models/backfill/561a_timepfn_synthetic_prior_scaled/best_model.pt`
- Adaptation windows: `441`
- Epochs: `8`
- Best adaptation loss: `0.7845`

Official-size 11-suite:

- Result: `results/autoresearch/561a_timepfn_synthetic_prior_scaled/full11.json`
- Score: `4/11`
- Passed: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`
- Failed: `coverage`, `conditionality`, `time_series`, `regime_coverage`, `distributional_fidelity`, `mean_reversion`, `pathwise_jump_realism`

## Key Metrics

- Overall coverage90: `0.888`
- h30 worst-cell coverage: `0.646`
- Conditionality MAE reduction: `3.1%`
- Time-series ACF correlation: `0.951`
- Tail-scale pass cells: `18/25`
- Cointegration gen/GT ratio: `0.869`
- Cointegration worst-cell ratio: `0.263`
- Regime layer2: `0/8`
- Daily-change KS pass cells: `10/25`
- Level KS pass cells: `2/25`
- Cross-cell correlation ratio: `0.901`
- Effective-rank ratio: `1.965`
- Mean-reversion aggregate ratio: `0.444`
- Pathwise max-jump KS: `0.911`

## Mechanism Read

The scaled TimePFN-style synthetic prior did improve basic support and dependence relative to the 560a smoke run. Surface validity, block-AR smoothness, cointegration, and cross-cell correlation all passed.

It still falls far below the `510a` risk prototype because the model is not sufficiently real-data adapted:

1. conditionality is below gate (`3.1%` MAE reduction);
2. h30 worst-cell coverage remains under the lower inclusion bar (`0.646`);
3. regime layer2 remains `0/8`;
4. mean reversion is too weak (`0.444`);
5. pathwise max-jump distribution is too far from validation (`KS=0.911`);
6. daily-change and level occupancy remain poor.

The key difference from `392a/510a` is training path, not architecture. `392a/510a` inherit a full real-data learned transition law and then receive recent-window objective tuning. 561a used synthetic pretraining followed only by the 441 recent adaptation window. That is too little real data to recover the empirical IV path law.

## Decision

Close "synthetic pretrain plus recent-only adaptation" as below frontier. Keep the broader TimePFN-style hypothesis alive only in the more faithful form:

1. synthetic prior pretraining;
2. broad real-data fine-tuning over the full pre-validation training history;
3. recent-window adaptation / patch-energy only after the real-data law is recovered.

The next HEAD iteration should run 562a by reusing the 561a synthetic checkpoint but fine-tuning on a much larger real window set before evaluating. This tests whether synthetic pretraining can help after the model relearns real IV dynamics, rather than expecting the recent 441 windows to do all adaptation.
