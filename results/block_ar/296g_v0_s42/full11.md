# 220h Full 11-Suite Multi-Horizon Validation

- model: `296g`
- checkpoint: `models/backfill/296g_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `surface, coverage, conditionality, time_series, regime_coverage, distributional_fidelity, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.845`
- h30 cov90: `0.953`
- turb/calm ratio: `1.106`
- MR ratio h1: `1.105`
- MR ratio h30: `0.754`

**Additional v2 Suites**
- time-series ACF corr: `0.907`
- block boundary ratio: `0.939`
- cointegration gen/GT ratio: `0.731`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `17/25`
- level KS pass cells: `0/25`
- corr ratio: `0.771`
- rank ratio: `1.913`
- max-jump KS: `0.388`
