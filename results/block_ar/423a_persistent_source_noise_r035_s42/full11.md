# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/423a_persistent_source_noise_r035_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, cointegration, regime_coverage, distributional_fidelity, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.858`
- h30 cov90: `0.983`
- turb/calm ratio: `1.044`
- MR ratio h1: `0.949`
- MR ratio h30: `0.700`

**Additional v2 Suites**
- time-series ACF corr: `0.941`
- block boundary ratio: `0.921`
- cointegration gen/GT ratio: `0.576`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `0/25`
- corr ratio: `0.923`
- rank ratio: `1.748`
- max-jump KS: `0.748`
