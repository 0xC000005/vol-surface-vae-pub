# 220h Full 11-Suite Multi-Horizon Validation

- model: `286b`
- checkpoint: `models/backfill/283a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.755`
- h30 cov90: `0.659`
- turb/calm ratio: `1.098`
- MR ratio h1: `1.359`
- MR ratio h30: `0.932`

**Additional v2 Suites**
- time-series ACF corr: `0.955`
- block boundary ratio: `1.098`
- cointegration gen/GT ratio: `0.816`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `23/25`
- level KS pass cells: `0/25`
- corr ratio: `0.987`
- rank ratio: `1.160`
- max-jump KS: `0.459`
