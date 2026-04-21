# 220h Full 11-Suite Multi-Horizon Validation

- model: `261c`
- checkpoint: `models/backfill/261c_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.668`
- h30 cov90: `0.813`
- turb/calm ratio: `0.924`
- MR ratio h1: `0.548`
- MR ratio h30: `0.767`

**Additional v2 Suites**
- time-series ACF corr: `0.950`
- block boundary ratio: `0.998`
- cointegration gen/GT ratio: `0.550`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `20/25`
- level KS pass cells: `16/25`
- corr ratio: `1.486`
- rank ratio: `0.519`
- max-jump KS: `0.968`
