# 220h Full 11-Suite Multi-Horizon Validation

- model: `261a`
- checkpoint: `models/backfill/261a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.832`
- h30 cov90: `0.972`
- turb/calm ratio: `1.119`
- MR ratio h1: `0.542`
- MR ratio h30: `0.687`

**Additional v2 Suites**
- time-series ACF corr: `0.944`
- block boundary ratio: `0.969`
- cointegration gen/GT ratio: `0.703`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `6/25`
- corr ratio: `0.980`
- rank ratio: `1.342`
- max-jump KS: `0.532`
