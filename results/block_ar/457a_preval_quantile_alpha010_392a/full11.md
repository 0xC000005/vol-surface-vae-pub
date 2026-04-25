# 403a Calibrated Risk-System Validation

- base model: `340c`
- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- calibration: `pre-validation empirical quantile map, regime_bins=True, alpha=0.1`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.850`
- h30 cov90: `0.877`
- turb/calm ratio: `1.062`

**Additional v2 Suites**
- time-series ACF corr: `0.954`
- block boundary ratio: `1.013`
- cointegration gen/GT ratio: `0.634`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `12/25`
- corr ratio: `0.958`
- rank ratio: `1.503`
- max-jump KS: `0.384`
