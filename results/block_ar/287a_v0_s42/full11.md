# 220h Full 11-Suite Multi-Horizon Validation

- model: `287a`
- checkpoint: `models/backfill/287a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.000`
- h30 cov90: `0.000`
- turb/calm ratio: `1.000`
- MR ratio h1: `1.741`
- MR ratio h30: `1.019`

**Additional v2 Suites**
- time-series ACF corr: `0.898`
- block boundary ratio: `1.213`
- cointegration gen/GT ratio: `1.247`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `20/25`
- level KS pass cells: `1/25`
- corr ratio: `1.012`
- rank ratio: `1.181`
- max-jump KS: `0.578`
