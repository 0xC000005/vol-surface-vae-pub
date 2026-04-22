# 220h Full 11-Suite Multi-Horizon Validation

- model: `286a`
- checkpoint: `models/backfill/283a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.766`
- h30 cov90: `0.677`
- turb/calm ratio: `1.039`
- MR ratio h1: `1.341`
- MR ratio h30: `1.056`

**Additional v2 Suites**
- time-series ACF corr: `0.952`
- block boundary ratio: `0.995`
- cointegration gen/GT ratio: `0.800`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `24/25`
- level KS pass cells: `0/25`
- corr ratio: `1.007`
- rank ratio: `1.200`
- max-jump KS: `0.306`
