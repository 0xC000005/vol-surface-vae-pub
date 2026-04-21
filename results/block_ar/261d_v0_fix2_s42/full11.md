# 220h Full 11-Suite Multi-Horizon Validation

- model: `261d`
- checkpoint: `models/backfill/261d_v0_fix2_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.675`
- h30 cov90: `0.859`
- turb/calm ratio: `1.021`
- MR ratio h1: `0.558`
- MR ratio h30: `0.755`

**Additional v2 Suites**
- time-series ACF corr: `0.947`
- block boundary ratio: `1.008`
- cointegration gen/GT ratio: `0.452`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `20/25`
- level KS pass cells: `11/25`
- corr ratio: `1.400`
- rank ratio: `0.586`
- max-jump KS: `0.985`
