# 462a Center-Preserving Residual Scale Calibration

- model: `462a_center_residual_scale_392a`
- base checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- calibration windows: `441` recent pre-validation
- validation windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity`

**Calibration Fit**
- scale range: `0.650` to `1.350`
- scale mean: `1.085`
- calibration coverage mean: `0.833` -> `0.879`

**Key Metrics**
- coverage90: `0.896`
- conditionality MAE reduction: `4.86%`
- regime layer2: `1/8`
- daily-change KS pass cells: `25/25`
- level KS pass cells: `12/25`
- corr ratio: `0.942`
- mean-reversion active pass rate: `0.858`
- max-jump KS: `0.304`
