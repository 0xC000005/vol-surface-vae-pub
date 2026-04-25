# 403a Calibrated Risk-System Validation

- base model: `340c`
- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- calibration: `pre-validation empirical quantile map, regime_bins=True, alpha=0.25`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.848`
- h30 cov90: `0.893`
- turb/calm ratio: `1.061`

**Additional v2 Suites**
- time-series ACF corr: `0.958`
- block boundary ratio: `0.984`
- cointegration gen/GT ratio: `0.715`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `11/25`
- corr ratio: `0.956`
- rank ratio: `1.487`
- max-jump KS: `0.383`
