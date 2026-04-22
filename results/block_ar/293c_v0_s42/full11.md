# 220h Full 11-Suite Multi-Horizon Validation

- model: `293c`
- checkpoint: `models/backfill/293c_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.830`
- h30 cov90: `0.919`
- turb/calm ratio: `0.990`
- MR ratio h1: `0.057`
- MR ratio h30: `0.288`

**Additional v2 Suites**
- time-series ACF corr: `0.960`
- block boundary ratio: `1.016`
- cointegration gen/GT ratio: `0.610`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `23/25`
- level KS pass cells: `1/25`
- corr ratio: `1.244`
- rank ratio: `0.881`
- max-jump KS: `0.674`
