# 220h Full 11-Suite Multi-Horizon Validation

- model: `263a`
- checkpoint: `models/backfill/263a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.753`
- h30 cov90: `0.865`
- turb/calm ratio: `0.874`
- MR ratio h1: `0.062`
- MR ratio h30: `0.259`

**Additional v2 Suites**
- time-series ACF corr: `0.946`
- block boundary ratio: `1.107`
- cointegration gen/GT ratio: `0.700`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `18/25`
- level KS pass cells: `3/25`
- corr ratio: `0.812`
- rank ratio: `1.192`
- max-jump KS: `0.820`
