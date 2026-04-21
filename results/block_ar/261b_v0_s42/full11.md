# 220h Full 11-Suite Multi-Horizon Validation

- model: `261b`
- checkpoint: `models/backfill/261b_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.091`
- h30 cov90: `0.101`
- turb/calm ratio: `0.853`
- MR ratio h1: `0.500`
- MR ratio h30: `0.694`

**Additional v2 Suites**
- time-series ACF corr: `0.644`
- block boundary ratio: `0.654`
- cointegration gen/GT ratio: `0.108`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `13/25`
- corr ratio: `1.225`
- rank ratio: `0.786`
- max-jump KS: `1.000`
