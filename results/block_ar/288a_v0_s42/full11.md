# 220h Full 11-Suite Multi-Horizon Validation

- model: `288a`
- checkpoint: `models/backfill/287e_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.000`
- h30 cov90: `0.000`
- turb/calm ratio: `1.000`
- MR ratio h1: `0.835`
- MR ratio h30: `0.711`

**Additional v2 Suites**
- time-series ACF corr: `0.951`
- block boundary ratio: `0.942`
- cointegration gen/GT ratio: `1.115`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `8/25`
- level KS pass cells: `20/25`
- corr ratio: `0.912`
- rank ratio: `1.413`
- max-jump KS: `0.766`
