# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/444a_recent_quantile_coverage_pathlaw_w2_s8_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `5/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.865`
- h30 cov90: `0.868`
- turb/calm ratio: `1.117`
- MR ratio h1: `0.911`
- MR ratio h30: `0.725`

**Additional v2 Suites**
- time-series ACF corr: `0.955`
- block boundary ratio: `0.965`
- cointegration gen/GT ratio: `0.674`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `4/25`
- corr ratio: `0.977`
- rank ratio: `1.481`
- max-jump KS: `0.318`
