# 462a Center-Preserving Residual Scale Calibration

- model: `462a_center_residual_scale_392a`
- base checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- calibration windows: `441` recent pre-validation
- validation windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity`

**Calibration Fit**
- scale range: `0.575` to `1.600`
- scale mean: `1.097`
- calibration coverage mean: `0.831` -> `0.879`

**Key Metrics**
- coverage90: `0.897`
- conditionality MAE reduction: `4.85%`
- regime layer2: `1/8`
- daily-change KS pass cells: `25/25`
- level KS pass cells: `13/25`
- corr ratio: `0.949`
- mean-reversion active pass rate: `0.858`
- max-jump KS: `0.287`
