# 220h Full 11-Suite Multi-Horizon Validation

- model: `287d`
- checkpoint: `models/backfill/287d_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.000`
- h30 cov90: `0.000`
- turb/calm ratio: `1.000`
- MR ratio h1: `0.168`
- MR ratio h30: `0.476`

**Additional v2 Suites**
- time-series ACF corr: `0.934`
- block boundary ratio: `1.027`
- cointegration gen/GT ratio: `1.273`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `24/25`
- level KS pass cells: `14/25`
- corr ratio: `0.907`
- rank ratio: `1.449`
- max-jump KS: `0.354`
