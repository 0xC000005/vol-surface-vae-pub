# 220h Full 11-Suite Multi-Horizon Validation

- model: `287b`
- checkpoint: `models/backfill/287b_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.000`
- h30 cov90: `0.000`
- turb/calm ratio: `1.000`
- MR ratio h1: `0.927`
- MR ratio h30: `0.912`

**Additional v2 Suites**
- time-series ACF corr: `0.945`
- block boundary ratio: `1.001`
- cointegration gen/GT ratio: `0.742`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `3/25`
- corr ratio: `1.001`
- rank ratio: `1.153`
- max-jump KS: `0.229`
