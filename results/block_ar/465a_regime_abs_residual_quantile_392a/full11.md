# 462a Center-Preserving Residual Scale Calibration

- model: `462a_center_residual_scale_392a`
- base checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- calibration windows: `441` recent pre-validation
- validation windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity`

**Calibration Fit**
- mode: `abs_quantile`
- scale range: `nan` to `nan`
- scale mean: `nan`
- calibration coverage mean: `nan` -> `nan`

**Key Metrics**
- coverage90: `0.882`
- conditionality MAE reduction: `4.91%`
- regime layer2: `1/8`
- daily-change KS pass cells: `25/25`
- level KS pass cells: `11/25`
- corr ratio: `0.914`
- mean-reversion active pass rate: `0.868`
- max-jump KS: `0.300`
