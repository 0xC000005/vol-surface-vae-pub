# 220h Full 11-Suite Multi-Horizon Validation

- model: `470a`
- checkpoint: `models/backfill/470a_frozen_center_residual_score_flow_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.892`
- h30 cov90: `0.730`
- turb/calm ratio: `0.898`
- MR ratio h1: `1.249`
- MR ratio h30: `0.822`

**Additional v2 Suites**
- time-series ACF corr: `0.900`
- block boundary ratio: `0.949`
- cointegration gen/GT ratio: `1.194`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `7/25`
- level KS pass cells: `11/25`
- corr ratio: `0.018`
- rank ratio: `4.378`
- max-jump KS: `0.536`
